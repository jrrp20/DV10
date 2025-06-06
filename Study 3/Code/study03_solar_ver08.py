from rdflib import Graph, RDFS, URIRef, Literal
from collections import defaultdict
import textwrap
from rdflib.namespace import RDF, OWL, split_uri
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
import smtplib
from email.message import EmailMessage
import os

def get_safe_max_new_tokens(prompt_text, model_id, context_window=8192, buffer=50):
    """
    Returns a safe max_new_tokens value so the total tokens (prompt + output)
    do not exceed the model's context window.

    Parameters:
    - prompt_text: str, the input prompt
    - model_id: str, the HF model identifier
    - context_window: int, max tokens the model supports (default 4096)
    - buffer: int, optional reserve tokens (for EOS etc.)

    Returns:
    - int: safe max_new_tokens value
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
    available_tokens = context_window - prompt_tokens - buffer

    if available_tokens < 0:
        print(f"⚠️ WARNING: Prompt exceeds context window by {-available_tokens} tokens.")
        return 0
    else:
        print(f"✅ Prompt uses {prompt_tokens} tokens. {available_tokens} tokens available for generation.")
        return available_tokens

# Load model and tokenizer
model_id = "upstage/SOLAR-0-70b-16bit"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",  # Ensures best dtype is used automatically
    use_safetensors=True,
    trust_remote_code=True,
    rope_scaling={"type": "dynamic", "factor": 2}  # enables longer inputs
)

# Create generation pipeline
generator = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    device_map="auto",
)

# final_samples input
final_samples = {'http://example.org/src#samplingtime': ['12:49:00'],
 'http://example.org/src#patient_cpr': ['te64687489', 'c0cef4fadfd', 'dc44b505e4e', 'afedd9d7f0', 'cdse4751d0'],
 'http://example.org/src#analysiscode': ['DNK35312', 'NPU02070'],
 'http://example.org/src#laboratorium_idcode': ['UKN','OUI','ESB','HDI','KPL'],
 'http://example.org/src#referenceinterval_lowerlimit': ['50.0', '137.0','27.0'],
 'http://example.org/src#referenceinterval_upperlimit': ['30.0','7.5','105.0'],
 'http://example.org/src#unit': ['U/L', 'mL/min', '10^6/l', 'mg/g', 'mol/l'],
 'http://example.org/src#rekvirent_idtype': ['sorkode','sygehusafdelingsnummer','yaugethusgbdnummer','ydernummer'],
 'http://example.org/src#samplingdate': ['2010-12-07','2017-04-16','2023-10-27'],
 'http://example.org/src#resulttype': ['alfanumerisk', 'numerisk'],
 'http://example.org/src#value': ['00', 'A RhD pos', '>175', 'NEG', '137'],
 'http://example.org/src#operator': ['stoerre_end', 'mindre_end'],
 'http://example.org/src#resultvalidation': ['for_hoej'],
 'http://example.org/src#rekvirent_id': ['0123815','1789AFS4611','2000A005','6620378SKADE','990202']}


# RDF graph of the source and target ontologies
src_graph = Graph().parse("flat_src_onto.owl")
tgt_graph = Graph().parse("flat_tgt_onto.owl")

# Helper to compact URIs to src: or tgt: style
def compact_uri(uri):
    uri_str = str(uri)
    if uri_str.startswith("http://example.org/src#"):
        return f"src:{uri_str.split('#')[-1]}"
    elif uri_str.startswith("http://example.org/tgt#"):
        return f"tgt:{uri_str.split('#')[-1]}"
    elif uri_str.startswith("http://www.w3.org/2001/XMLSchema#"):
        return f"xsd:{uri_str.split('#')[-1]}"
    else:
        return uri_str

def generate_src_domain_description_with_classes(graph, final_samples, minimal=False, include_examples=False):
    lines = ["### Source Ontology Description (`src:`)\n"]

    # --- Add class descriptions ---
    lines.append("#### Classes\n")
    for cls in graph.subjects(RDF.type, OWL.Class):
        lines.append(f"- {compact_uri(cls)}")
    lines.append("")

    # --- Add property descriptions ---
    lines.append("#### Properties\n")
    for uri_str, examples in final_samples.items():
        uri = URIRef(uri_str)
        domain = graph.value(uri, RDFS.domain)
        range_ = graph.value(uri, RDFS.range)

        lines.append(f"- {compact_uri(uri)}")
        if domain:
            lines.append(f"  - Domain: {compact_uri(domain)}")
        if range_:
            lines.append(f"  - Range: {compact_uri(range_)}")

        if not minimal:
            label = graph.value(uri, RDFS.label)
            comment = graph.value(uri, RDFS.comment)
            if label:
                lines.append(f"  - Label: \"{str(label)}\"")
            if comment:
                lines.append(f"  - Comment: \"{str(comment)}\"")
            if include_examples and examples:
                example_line = textwrap.fill(', '.join(examples), width=80,
                                             initial_indent='  - Example values: ',
                                             subsequent_indent=' ' * 20)
                lines.append(example_line)
        lines.append("")

    return '\n'.join(lines)

def generate_tgt_domain_description_with_classes(graph, minimal=False):
    lines = ["### Target Ontology Description (`tgt:`)\n"]

    # --- Add class descriptions ---
    lines.append("#### Classes\n")
    for cls in graph.subjects(RDF.type, OWL.Class):
        lines.append(f"- {compact_uri(cls)}")
    lines.append("")

    # --- Add property descriptions ---
    lines.append("#### Properties\n")
    properties = set(graph.subjects(RDFS.domain, None)) | set(graph.subjects(RDFS.range, None))

    for uri in sorted(properties):
        domain = graph.value(uri, RDFS.domain)
        range_ = graph.value(uri, RDFS.range)

        lines.append(f"- {compact_uri(uri)}")
        if domain:
            lines.append(f"  - Domain: {compact_uri(domain)}")
        if range_:
            lines.append(f"  - Range: {compact_uri(range_)}")

        if not minimal:
            label = graph.value(uri, RDFS.label)
            comment = graph.value(uri, RDFS.comment)
            if label:
                lines.append(f"  - Label: \"{str(label)}\"")
            if comment:
                lines.append(f"  - Comment: \"{str(comment)}\"")
        lines.append("")

    return '\n'.join(lines)

# Generate descriptions including examples for the source
src_domain_str = generate_src_domain_description_with_classes(src_graph, final_samples, minimal=False, include_examples=True)
tgt_domain_str = generate_tgt_domain_description_with_classes(tgt_graph, minimal=False)

minimal_src_domain_str = generate_src_domain_description_with_classes(src_graph, final_samples, minimal=True, include_examples=False)
minimal_tgt_domain_str = generate_tgt_domain_description_with_classes(tgt_graph, minimal=True)

src_domain_str_wo_examples = generate_src_domain_description_with_classes(src_graph, final_samples, minimal=False, include_examples=False)
######################################################################################
# Step 1: Ontology Matching
######################################################################################
matching_prompt = f"""
## Task: Ontology Matching

You are an expert in ontology matching.

Given a **Source Ontology** (`src:`) and a **Target Ontology** (`tgt:`), identify semantic matches: corresponding classes and properties between the ontologies. Your task is to find and list all **semantic matches**—pairs of concepts that refer to the same or closely related ideas.

---

### 🧠 What to Match
- Match classes or properties with the **same or closely related meaning**
- Use similarities in **labels, comments, domains, ranges, and names**
- Include only clear, meaningful correspondences

- **Matchings may include**:
  - 1:1 — One source concept ≈ One target concept  
  - 1:N — One source concept ≈ Multiple target concepts  
  - N:1 — Multiple source concepts ≈ One target concept  
  - M:N — Multiple source concepts ≈ Multiple target concepts  

- When more than one match exists for a concept, include **each match on its own line**
- If no suitable target exists for a source concept, **omit it from the output**

---

### 🧮 Matching Procedure

Iterate systematically:

- For each `src:` concept in the Source Ontology:
  - Compare it to each `tgt:` concept in the Target Ontology.
  - If a semantic match is found, output a line in this format:
    `src:ConceptA ≈ tgt:ConceptB`
  - If multiple matches exist for a concept, output one line per match.
  - If no match exists, skip the concept.

---

### 🔍 What Qualifies as a Semantic Match
A **semantic match** exists when two concepts refer to the same or equivalent meaning — even if they differ in structure or representation. Valid matches include:
- One concept being a **component** of another (e.g., `src:firstname` vs. `tgt:fullname`)
- Cases where **converting**, **parsing**, or **value transformation** would be required
- Multiple target concepts jointly representing a source concept

You are not required to define the transformation — only to identify that a match exists.

---

#### ✅ Examples of Valid Matches
- src:birthDate ≈ tgt:date_of_birth  
- src:birthDate ≈ tgt:personAge # Requires calculating age from birth date
- src:height_m ≈ tgt:personHeight_feet # Requires unit conversion (meters → feet)
- src:employmentStatus ≈ tgt:employment_status # Requires a lookup: src: uses integers (e.g. 1, 2), tgt: uses strings (e.g. "employed", "unemployed")

---

### 🗂️ Ontologies

**Source Ontology** (`src:`):  
{src_domain_str}

**Target Ontology** (`tgt:`):  
{tgt_domain_str}

---

### 📤 Output Format

Each line must contain:
- A `src:` element
- The ≈ symbol
- A `tgt:` element (or ∅ if unmatched)
 
❗ Output only a list of matched pairs using this exact format:
- src:PropertyA ≈ tgt:PropertyB  
- src:ClassX ≈ tgt:ClassY  

✅ End your response **after you have iterated over all concepts in the Source Ontology**.

Begin now:
"""

safe_tokens_matching_task = get_safe_max_new_tokens(matching_prompt, model_id)

# Run alignment generation with timing
start_time = time.time()

matching_output = generator(
    matching_prompt,
    max_new_tokens=safe_tokens_matching_task,
    do_sample=False,
    temperature=0.1,
    top_p=0.6,
    return_full_text=False,  # optional, cleaner output
)[0]['generated_text']

end_time = time.time()

matching_timing = end_time - start_time

print(f"\nLLM response time (Matchings Generation): {matching_timing:.2f} seconds\n")
print(matching_output)

######################################################################################
# Step 2: Ontology Mapping
######################################################################################
alignment_prompt = f"""
## Task: Ontology Alignment via First-Order Logic (FOL)

You are a knowledge representation expert.

Given a source ontology and a target ontology, produce a complete set of logical alignments between them. These alignments describe how to semantically transform data from the source to the target. Express all mappings using first-order logic (FOL).

---

### 🗂️ Ontology Context

The source ontology includes example values for its datatype properties. The target ontology includes structural definitions only.

**Source Ontology** (`src:`):  
{src_domain_str}

**Target Ontology** (`tgt:`):  
{tgt_domain_str}

---

### ✅ Requirements:

#### 🔗 Matched Elements

Only generate mappings for the following matched properties and classes:
{matching_output}

---

#### 🔄 Mapping Relationships

- Mappings may include:
  - 1:1 mappings (direct correspondence)
  - 1:2 or 2:1 mappings (split or combine)
  - n:m mappings (many-to-many relationships)
- Mappings may involve intermediate transformation steps such as:
  - Concatenating multiple values
  - Decomposing or parsing a value into structured parts
  - Mapping string or code values to concept identifiers via lookup tables
- In addition to properties, identify and align source and target classes.
- When instances of a source class (e.g., `src:ClassA`) should be transformed into instances of a corresponding target class (e.g., `tgt:ClassB`), include this as a class-level mapping:

  ∀x (src:ClassA(x) → tgt:ClassB(x))

---

#### 🔧 Parsing Literal Values

- When a literal value must be parsed into multiple structured components (e.g., extracting an operator and a number from a threshold string), assume a helper function exists to do so.
- Select an appropriate name for the parsing function using the format:

  `parse_{{value_type}}` or `parse_{{semantic_task}}`

- Examples (naming conventions only):
  - `parse_threshold`
  - `parse_date_range`
  - `parse_value_with_unit`

- These parsing functions are to be treated as opaque and deterministic. Use them directly in the logic expressions.

---

#### 📚 Lookup Functions for Concept Resolution

- When values such as codes, units, or symbolic strings need to be mapped to concept identifiers in the target ontology, assume the existence of a lookup function.
- Name the function according to the target resource it returns:

  `lookup_{{target_resource_name}}(input_value)`

- Examples (naming conventions only):
  - `lookup_concept_id`
  - `lookup_unit_concept`
  - `lookup_operator_concept`

- These functions return target concept identifiers and can be used directly in logical rules.

---

### 📘 Illustrative Examples (Structure Only)

The examples below demonstrate the structure of valid alignment rules and usage of helper functions. They are not tied to any specific ontology or domain.

∀x,v (src:hasValue(x,v) ∧ parse_threshold(v,op,n) → tgt:operatorConceptId(x, lookup_operator_concept(op)) ∧ tgt:valueAsNumber(x,n))

∀x,c (src:hasCode(x,c) → tgt:conceptId(x, lookup_concept_id(c)))

---

### 📤 Output Format

- Produce a list of distinct, universally quantified first-order logic (FOL) rules that define how source predicates map to target predicates.
- Follow this structure:

  ∀x,y,... (src:Triple1 ∧ src:Triple2 → tgt:Triple3 ∧ tgt:Triple4 ...)

- Use predicates of the form `src:propertyName` and `tgt:propertyName`.
- Each rule must be:
  - Syntactically valid
  - Semantically meaningful
  - Logically sound
"""

safe_tokens_alignm_task = get_safe_max_new_tokens(alignment_prompt, model_id)

# Run alignment generation with timing
start_time = time.time()

alignment_output = generator(
    alignment_prompt,
    max_new_tokens=safe_tokens_alignm_task,
    do_sample=False,
    temperature=0.1,
    top_p=0.6,
    return_full_text=False,  # optional, cleaner output
)[0]['generated_text']

end_time = time.time()

alignment_timing = end_time - start_time

print(f"\nLLM response time (Mappings Generation): {alignment_timing:.2f} seconds\n")
print(alignment_output)

######################################################################################
# Step 3: FGF Generation
######################################################################################

fgf_prompt = f"""You are an expert Python developer specializing in RDF transformations.

Your task is to generate **Fact Generating Functions (FGFs)** — Python functions that use RDFLib to transform RDF triples from a source graph (`src_graph`) into a target graph (`tgt_graph`) based on a formal ontology alignment.

---

### 🔁 Definition of FGFs

Each FGF:
- Handles a specific **source class** (e.g., `src:Person`).
- Iterates over all instances of that class in `src_graph`.
- Creates a corresponding instance in `tgt_graph` using the mapped target class.
- Transfers property values using explicitly defined mappings.
- Skips any information that is not mapped.
- Uses the `SRC` and `TGT` namespaces to build URIs.
- Ensures that domain and range constraints of the **target ontology** are respected.
- Follows RDFLib syntax and best practices.

---

### 📑 Ontology Mappings

The `mappings_block` below consists of first-order logic (FOL) alignment rules describing how source classes and properties map to target ontology structures.

```text
{alignment_output}
```

#### Ontology Domain and Range (for validation)

**Source Ontology** (`src:`):
{minimal_src_domain_str}

## Target ontology (`tgt:`):
{minimal_tgt_domain_str}

### ✅ Instructions

For each source class that appears as the subject of a class-level alignment:
1. Write a single Python function that:
   - Finds all instances of the source class in `src_graph`.
   - Constructs a new instance of the mapped target class in `tgt_graph`.
   - Transfers mapped properties using the logic in mappings_block.
2. Use RDFLib's `URIRef`, `Literal`, and `RDF.type` as appropriate.
3. Construct URIs using `SRC` and `TGT` namespaces.
4. Do not include unmapped properties or relationships.
5. Ensure all triples conform to the target ontology's domain and range constraints.
6. You may call helper functions like parse_threshold() or lookup_concept_id() if referenced in the mappings.
7. Each function must be standalone, valid Python, and directly executable in a transformation pipeline.

---

### ⚠️ Output Constraint

Only output the Python function(s) corresponding to the mapped classes.   
Output code only.
"""

safe_tokens_fgf_task = get_safe_max_new_tokens(fgf_prompt, model_id)

# Run generation with timing
start_time = time.time()

fgf_output = generator(
    fgf_prompt,
    max_new_tokens=safe_tokens_fgf_task,
    do_sample=False,
    temperature=0.1,
    top_p=0.6,
    return_full_text=False,  # optional, cleaner output
)[0]['generated_text']

end_time = time.time()
fgf_timing = end_time - start_time

print(f"\nLLM response time (FGF Generation): {fgf_timing:.2f} seconds\n")
print(fgf_output)