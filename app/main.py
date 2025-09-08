# app_assess_copa_3320010.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os, json
from dotenv import load_dotenv
# ---- Env setup ----

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

app = FastAPI(title="SAP Note 3320010 CO-PA Type Change Assessment")
# --- SNIPPET HELPER ---
def snippet_at(text: str, start: int, end: int) -> str:
    s = max(0, start - 60)
    e = min(len(text), end + 60)
    return text[s:e].replace("\n", "\\n")

# ===== Models =====
class CopaUsage(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: str
    snippet: Optional[str] = None   # 👈 added snippet support

class Unit(BaseModel):
    pgm_name: str
    inc_name: str
    type: str
    name: str
    copa_usage: List[CopaUsage] = Field(default_factory=list)

# ===== Summariser for SAP Note 3320010 =====
def summarize_copa(unit: Unit) -> Dict[str, Any]:
    """
    Checks for:
    1) IS INITIAL / IS NOT INITIAL on RKEOBJNR or ProfitabilitySegment
    2) Usage of decommissioned CDS field ProfitabilitySegment (should be ProfitabilitySegment_2)
    """
    flagged = []

    for usage in unit.copa_usage:
        stmt_upper = (usage.suggested_statement or "").upper()
        # Check for initial value checks
        if "IS INITIAL" in stmt_upper or "IS NOT INITIAL" in stmt_upper:
            if "RKEOBJNR" in stmt_upper or "PROFITABILITYSEGMENT" in stmt_upper:
                flagged.append({
                    "issue": "Initial value check on RKEOBJNR/ProfitabilitySegment",
                    "reason": "Must replace with cl_fco_copa_paobjnr=>is_initial( ) per SAP Note 3320010"
                })
        # Check for old CDS field name
        if "PROFITABILITYSEGMENT" in stmt_upper and "PROFITABILITYSEGMENT_2" not in stmt_upper:
            flagged.append({
                "issue": "Old CDS field ProfitabilitySegment used",
                "reason": "Replace with ProfitabilitySegment_2 per SAP Note 3320010"
            })

    return {
        "program": unit.pgm_name,
        "include": unit.inc_name,
        "unit_type": unit.type,
        "unit_name": unit.name,
        "flags": flagged
    }

# ===== Prompt for CO-PA RKEOBJNR fix =====
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 3320010. Output strict JSON only."

USER_TEMPLATE = """
You are assessing ABAP code for CO-PA (Profitability Analysis) changes per SAP Note 3320010.

Summary of technical impact:
- Profitability segment number field RKEOBJNR changed from NUMC(10) to CHAR(10).
- IS INITIAL / IS NOT INITIAL checks no longer valid; must use cl_fco_copa_paobjnr=>is_initial( ).
- CDS field ProfitabilitySegment is decommissioned; must use ProfitabilitySegment_2.
- Must handle both old ('0000000000') and new (spaces) initial values.
- Code should be future-proof for alphanumeric segment numbers.

Your tasks:
1) Produce a concise **assessment** of the given unit's code.
2) Produce an **LLM remediation prompt** that:
   - Searches code for IS INITIAL/IS NOT INITIAL checks on RKEOBJNR or ProfitabilitySegment.
   - Replaces them with cl_fco_copa_paobjnr=>is_initial( ) calls.
   - Replaces CDS field ProfitabilitySegment with ProfitabilitySegment_2.
   - Handles both old and new initial values.
   - Keeps functional behavior unchanged.

Return ONLY strict JSON:
{{
  "assessment": "<concise note 3320010 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {unit_type}
- Unit name: {unit_name}

Analysis:
{plan_json}

copa_usage (JSON):
{copa_json}
""".strip()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG),
        ("user", USER_TEMPLATE),
    ]
)

llm = ChatOpenAI(model=OPENAI_MODEL)
parser = JsonOutputParser()
chain = prompt | llm | parser

# ===== LLM Call =====
def llm_assess_and_prompt(unit: Unit) -> Dict[str, str]:
    plan = summarize_copa(unit)
    plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
    copa_json = json.dumps([s.model_dump() for s in unit.copa_usage], ensure_ascii=False, indent=2)

    try:
        return chain.invoke({
            "pgm_name": unit.pgm_name,
            "inc_name": unit.inc_name,
            "unit_type": unit.type,
            "unit_name": unit.name,
            "plan_json": plan_json,
            "copa_json": copa_json
        })
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

# ===== API =====
@app.post("/assess-copa-3320010")
async def assess_copa(units: List[Unit]) -> List[Dict[str, Any]]:
    out = []
    for u in units:
        obj = u.model_dump()
        llm_out = llm_assess_and_prompt(u)
        obj["assessment"] = llm_out.get("assessment", "")
        obj["llm_prompt"] = llm_out.get("llm_prompt", "")
        obj.pop("copa_usage", None)
        out.append(obj)
    return out

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
