from rdflib import Graph, URIRef, Literal, OWL, RDF, RDFS, SKOS
from urllib.parse import urlparse
import re
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import time
from collections import defaultdict


def extract_text(graph, resource):
    texts = set()
    label_props = [RDFS.label, SKOS.prefLabel, URIRef('http://schema.org/name')]
    desc_props = [RDFS.comment, URIRef('http://purl.org/dc/terms/description'), URIRef('http://schema.org/comment')]

    for prop in label_props + desc_props:
        for obj in graph.objects(resource, prop):
            if isinstance(obj, Literal):
                texts.add(str(obj).strip())

    # Include URI fragment if useful
    uri_fragment = urlparse(str(resource)).fragment
    if uri_fragment and len(re.findall(r'\d', uri_fragment))/len(uri_fragment) < 0.5:
        texts.add(uri_fragment.strip())

    # Recursive annotations (only literals from annotations)
    for annotation_p, annotation_o in graph.predicate_objects(resource):
        if isinstance(annotation_o, URIRef) and annotation_o != resource:
            texts.update(extract_text(graph, annotation_o))

    # Remove empty strings or meaningless entries
    texts = {t for t in texts if len(t.strip()) > 2}

    return " ".join(sorted(texts))

def load_ontology_texts(ontology_file):
    g = Graph().parse(ontology_file)
    resource_texts = {}
    for res in set(g.subjects()):
        text = extract_text(g, res)
        if text:  # Only include if text extraction was successful
            resource_texts[str(res)] = text
    return resource_texts

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and embed source ontology
src_texts = load_ontology_texts("src_onto_test_two.owl")
src_embeddings = model.encode(list(src_texts.values()))

# Load and embed target ontology
tgt_texts = load_ontology_texts("tgt_onto_test_two.owl")
tgt_embeddings = model.encode(list(tgt_texts.values()))

# Initialize ChromaDB persistent client
client = chromadb.PersistentClient(path="./ontology_chromadb_sol_simp")

# Create collections
src_collection = client.get_or_create_collection(name="source")
tgt_collection = client.get_or_create_collection(name="target")

# Add to source collection
src_collection.add(
    embeddings=src_embeddings.tolist(),
    documents=list(src_texts.values()),
    metadatas=[{"uri": uri} for uri in src_texts.keys()],
    ids=list(src_texts.keys())
)

# Add to target collection
tgt_collection.add(
    embeddings=tgt_embeddings.tolist(),
    documents=list(tgt_texts.values()),
    metadatas=[{"uri": uri} for uri in tgt_texts.keys()],
    ids=list(tgt_texts.keys())
)

def match_ontology_entity_with_llm(
    src_uri,
    src_text,
    src_collection,
    tgt_collection,
    tokenizer,
    model,
    top_k=3,
    positive_words=["yes"],
    negative_words=["no"],
    confidence_threshold=0.6
):
    import torch
    import torch.nn.functional as F

    def classify_and_score_yes_no(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model(**inputs)
            next_token_logits = output.logits[0, -1]
        probs = F.softmax(next_token_logits, dim=-1)
        pos_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in positive_words]
        neg_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in negative_words]
        pos_prob = max(probs[i].item() for i in pos_ids)
        neg_prob = max(probs[i].item() for i in neg_ids)
        if pos_prob + neg_prob == 0:
            return 0.5, "uncertain"
        confidence = pos_prob / (pos_prob + neg_prob)
        return confidence, "yes" if confidence >= 0.5 else "no"

    # === Step 1: Find top-k candidates ===
    src_doc = src_collection.get(ids=[src_uri], include=['embeddings'])
    if not src_doc or len(src_doc['embeddings']) == 0:
        print(f"No embedding found for {src_uri}")
        return None

    query_embedding = src_doc['embeddings'][0]
    results = tgt_collection.query(query_embeddings=[query_embedding], n_results=top_k)
    candidates = zip(results['ids'][0], results['documents'][0])

    # === Step 2: Evaluate with LLM ===
    best_match = None
    best_score = 0.0

    for tgt_uri, tgt_text in candidates:
        prompt = f"""Answer with 'yes' or 'no':
Do the following two concepts refer to the same real-life thing?

Concept 1:
{src_text}

Concept 2:
{tgt_text}

Answer:"""

        confidence, prediction = classify_and_score_yes_no(prompt)
        print(f"{src_uri} vs {tgt_uri} → {prediction.upper()} (confidence: {confidence:.2f})")

        if prediction == "yes" and confidence > best_score:
            best_match = {"src": src_uri, "tgt": tgt_uri, "confidence": confidence}
            best_score = confidence

    # === Step 3: Return match if confident ===
    if best_match and best_match["confidence"] >= confidence_threshold:
        print(f"MATCH: {src_uri} ↔ {best_match['tgt']} (confidence: {best_score:.2f})")
        return best_match
    else:
        print(f"No confident match found for {src_uri}")
        return None
    
model_id = "upstage/SOLAR-0-70b-16bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model=AutoModelForCausalLM.from_pretrained(
     model_id,
     device_map="auto",
     use_safetensors=True,
 )

final_matches = {}

t1 = time.time()
for src_uri, src_text in src_texts.items():
    result = match_ontology_entity_with_llm(
        src_uri=src_uri,
        src_text=src_text,
        src_collection=src_collection,
        tgt_collection=tgt_collection,
        tokenizer=tokenizer,
        model=llm_model,
        top_k=3
    )
    if result:
        final_matches[result["src"]] = {
            "match": result["tgt"],
            "confidence": result["confidence"]
        }
time_find_final_matches = time.time()-t1

print(f"It took {time_find_final_matches} to find the final matches.")
print(final_matches)

def generate_domain_range_description(graph, prefix_label):
    from rdflib import RDF, RDFS, OWL, Literal
    from collections import defaultdict

    output_lines = []
    properties = defaultdict(dict)

    for s in graph.subjects(RDF.type, OWL.ObjectProperty):
        domain = next(graph.objects(s, RDFS.domain), None)
        range_ = next(graph.objects(s, RDFS.range), None)
        properties[s]["type"] = "object"
        properties[s]["domain"] = domain
        properties[s]["range"] = range_

    for s in graph.subjects(RDF.type, OWL.DatatypeProperty):
        domain = next(graph.objects(s, RDFS.domain), None)
        range_ = next(graph.objects(s, RDFS.range), None)
        properties[s]["type"] = "datatype"
        properties[s]["domain"] = domain
        properties[s]["range"] = range_ if range_ else Literal

    for prop, details in properties.items():
        domain_str = details["domain"].split("#")[-1] if details["domain"] else "None"
        range_str = (
            details["range"].split("#")[-1] if hasattr(details["range"], "split") else "Literal"
        )
        prop_label = prop.split("#")[-1]
        output_lines.append(f"- {prop_label}: domain=`{domain_str}`, range=`{range_str}`")

    return "\n".join(sorted(output_lines))

# Load OWL ontologies (adjust paths as needed)
src_graph = Graph().parse("src_onto_test_two.owl")
tgt_graph = Graph().parse("tgt_onto_test_two.owl")

# Generate the formatted domain/range description strings
src_domain_str = generate_domain_range_description(src_graph, "src")
tgt_domain_str = generate_domain_range_description(tgt_graph, "tgt")

def build_fgf_prompt(final_matches, src_domain_str, tgt_domain_str):
    # Convert final_matches to Python dict syntax (as string)
    mapping_lines = [f'    "{src}": "{match_info["match"]}"' for src, match_info in final_matches.items()]
    mappings_block = "mappings = {\n" + ",\n".join(mapping_lines) + "\n}"

    # Inject everything into the base prompt
    prompt = f"""You are an expert Python developer specializing in RDF transformations.

Your task is to generate **Fact Generating Functions (FGFs)** — Python functions that use RDFLib to transform RDF triples from a source graph (`src_graph`) into a target graph (`tgt_graph`) based on an ontology alignment.

---

### 🔁 Definition of FGFs

Each FGF:
- Handles a specific **source class** (e.g., `src:Person`).
- Iterates over all instances of that class in `src_graph`.
- Creates a corresponding instance in `tgt_graph` using the mapped target class.
- Copies property values using explicitly defined mappings.
- Skips any information that is not mapped.
- Uses the `SRC` and `TGT` namespaces to build URIs.
- Ensures that domain and range constraints of the **target ontology** are respected.
- Follows RDFLib syntax and best practices.

---

### 📑 Ontology Mappings

```python
{mappings_block}
```
#### Ontology Domain and Range (for validation)

**Source Ontology** (`src:`):
{src_domain_str}

## Target ontology (`tgt:`):
{tgt_domain_str}

### ✅ Instructions

For each mapped source class in the `mappings` dictionary:
1. Write a single Python function that:
   - Finds all instances of the source class in `src_graph`.
   - Constructs a new instance of the mapped target class in `tgt_graph`.
   - Transfers mapped properties using the mappings.
2. Use RDFLib's `URIRef`, `Literal`, and `RDF.type` as appropriate.
3. Construct URIs using `SRC` and `TGT` namespaces.
4. Do not include unmapped properties or relationships.
5. Ensure all triples conform to the target ontology's domain and range constraints.
6. Your code should be modular, clear, and ready to use in a transformation pipeline.

---

### ⚠️ Output Constraint

Only output the Python function(s) corresponding to the mapped classes.  
Do **not** include explanations, repeated mappings, examples, or comments.  
Output code only.
"""
    return prompt

prompt = build_fgf_prompt(final_matches, src_domain_str, tgt_domain_str)

generator = pipeline(
     "text-generation",
     model=llm_model,
     tokenizer=tokenizer,
     device_map="auto",
     do_sample=True,
     temperature=0.2,
     top_p=0.6,
)
# Time the generation
start_time = time.time()

output = generator(prompt, max_new_tokens=2048, temperature=0.1)[0]['generated_text']
end_time = time.time()
elapsed = end_time - start_time

print(f"\nLLM response time: {elapsed:.2f} seconds\n")
print(output)