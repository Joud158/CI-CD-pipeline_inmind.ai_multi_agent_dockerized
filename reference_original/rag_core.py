from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import faiss # for vector similarity search 
import numpy as np # numeric arrays 
from dotenv import load_dotenv # load env variables
from sentence_transformers import SentenceTransformer # embeddings

load_dotenv()

# agronomy text files 
DUMMY_DOCS: Dict[str, str] = {'01_crop_diagnostics.txt': '# Chapter 1 — Crop Diagnostics & Integrated Pest Management (IPM)\n\n## 1.1 Why diagnosis matters\nFarm decisions are expensive: spraying the wrong product wastes money, delays recovery, and can increase resistance. A good diagnosis starts with: (1) crop and variety, (2) growth stage, (3) pattern in the field (edges vs center), (4) weather last 7–14 days, (5) recent inputs (fertilizer, pesticide, irrigation).\n\n## 1.2 Fast field checklist (5 minutes)\n- Look at distribution: random spots often indicate disease; straight lines often indicate irrigation or equipment issues.\n- Check old vs new leaves: nutrient issues often show patterns (e.g., nitrogen on older leaves first).\n- Inspect underside of leaves for mites, eggs, and fungal growth.\n- Cut stems: brown vascular tissue can indicate wilts.\n- Smell and feel: bacterial soft rots often smell and feel watery.\n\n## 1.3 Powdery mildew on cucurbits (cucumber, zucchini, melon)\nTypical symptoms:\n- White, powder-like patches on the upper leaf surface; later, patches spread and leaves yellow and dry.\n- Often appears after warm days and cool nights; dense canopy and poor air flow increases risk.\n\nWhat it is:\n- A fungal disease; spores move by wind and can spread quickly.\n\nManagement (IPM):\n1) Cultural\n   - Increase spacing; prune to improve air movement.\n   - Avoid excess nitrogen late in the season (too much soft growth).\n   - Remove heavily infected leaves early; do not compost if infection is severe.\n2) Monitoring\n   - Scout twice per week once first symptoms are seen.\n   - Use a simple severity score: 0=no spots, 1=1–5 spots, 2=patches on <25% leaf, 3=25–50%, 4=>50%.\n3) Chemical (when justified)\n   - Rotate modes of action to reduce resistance. Do not repeat the same active ingredient back-to-back.\n   - Apply preventively when the first spots appear; late spraying gives poor results because the fungus is already established.\n4) Decision rule\n   - If severity ≥2 on young leaves and weather stays favorable, treat.\n   - If infection is only on very old leaves and harvest is near, focus on leaf removal and airflow.\n\n## 1.4 Early blight on tomato (and potato)\n- Concentric “target” rings on older leaves; leaf drop starts from the bottom.\n- Risk increases with leaf wetness (overhead irrigation) and warm humid conditions.\n- Reduce leaf wetness: drip irrigation, morning watering, staking, mulching.\n- Remove lower leaves touching soil; sanitize tools.\n\n## 1.5 Aphids and virus risk\nAphids cause direct damage (curling, sticky honeydew) and can transmit viruses. Virus symptoms can include mosaic patterns and stunting. Even low aphid numbers can be important if virus is present nearby.\n\nPractical actions:\n- Use yellow sticky cards as an early warning.\n- Control weeds around the plot (virus reservoirs).\n- If virus symptoms appear, remove infected plants early to protect the rest.\n\n## 1.6 “Spray or not?” logic\nBefore spraying, answer:\n- Is the problem biotic (pest/disease) or abiotic (nutrition, water, heat)?\n- What is the expected yield loss if you do nothing?\n- What is the cost of action (product + labor + fuel) and chance of success?\n- Will the action increase resistance risk?\n\nIf expected benefit < cost, do not spray. Instead, choose a lower-cost action (remove leaves, improve airflow, adjust irrigation) and monitor.\n\n## 1.7 Safe use notes (high-level)\nFollow label instructions, pre-harvest intervals, and protective equipment. Store chemicals away from children and food.', '02_irrigation_water.txt': '# Chapter 2 — Irrigation Scheduling, Soil, and Water Quality\n\n## 2.1 The goal\nIrrigation should keep water in the root zone without waterlogging or salt buildup. Too little water reduces growth; too much water causes root diseases, nutrient leaching, and wasted pumping cost.\n\n## 2.2 Soil texture changes everything\nSandy soils:\n- Low water-holding capacity; water moves fast.\n- Best practice: shorter, more frequent irrigations (e.g., daily or every 2 days in hot periods).\n\nLoamy soils:\n- Balanced holding capacity; moderate frequency.\n\nClay soils:\n- High holding, slow infiltration; longer but less frequent irrigation; avoid saturating because roots need oxygen.\n\n## 2.3 Practical scheduling without fancy tools\nUse a simple “bucket” view of the root zone:\n- Root depth (vegetables): often 20–40 cm early, 40–60 cm later.\n- Allowable depletion (rule of thumb): irrigate when ~30–50% of available water is used.\n\nField method:\n- Dig a small hole 20–30 cm deep.\n- Squeeze soil: if it forms a weak ball and crumbles, you’re near the irrigation threshold.\n- If it forms a sticky ribbon, it may be too wet.\n\n## 2.4 Crop stage matters (example: tomato)\n- After transplanting (first 7–10 days): frequent small irrigations to reduce stress.\n- Vegetative growth: steady moisture; avoid big swings.\n- Flowering/fruit set: water stress can reduce fruit set; prioritize consistent irrigation.\n- Ripening: slightly reduce water to limit cracking, but do not let plants wilt.\n\n## 2.5 Drip irrigation vs sprinkler\nDrip delivers water to roots and keeps leaves dry, reducing fungal disease (early blight, downy mildew). Sprinklers cool the canopy but increase leaf wetness and disease risk.\n\n## 2.6 Salinity and “white crust”\nHigh salinity water can leave white crust on soil surface. Signs include poor germination and leaf burn on edges. Management:\n- Use drip to concentrate water near roots.\n- Occasionally apply a leaching irrigation if drainage is adequate.\n- Mulch to reduce evaporation (evaporation leaves salts behind).\n\n## 2.7 Simple irrigation recipe for sandy soil vegetables\nIf you have sandy soil and drip lines:\n- Start with 20–30 minutes per day (or per irrigation cycle), then adjust using plant response.\n- In hot windy days, split into 2 cycles (morning + late afternoon) to reduce deep percolation loss.\n- Watch leaves at midday: mild midday droop can be normal; severe droop that continues into evening means under-watering.\n\n## 2.8 Weather notes for coastal Mediterranean spring\nIn spring, temperature swings are common. Wind can increase evapotranspiration even if temperature is moderate. After a heat spike, plants can suddenly need more water.\n\n## 2.9 Record keeping\nWrite down irrigation duration, date, and any rainfall. This helps calibrate your schedule to your field.', '03_soil_nutrition.txt': '# Chapter 3 — Soil Testing, Fertility, and Fertigation\n\n## 3.1 Start with a soil test\nA soil test prevents guessing. Key values:\n- pH (affects nutrient availability)\n- Organic matter (buffering and water holding)\n- Electrical conductivity (salinity)\n- N, P, K levels; plus micronutrients (Fe, Zn, B)\n\n## 3.2 Interpreting common patterns\n- Low organic matter: nutrients leach quickly; split applications work better.\n- High pH (alkaline): iron and zinc become less available; chlorosis can appear on young leaves.\n- Very high phosphorus: can reduce zinc uptake.\n\n## 3.3 N-P-K basics (simple view)\n- Nitrogen (N): vegetative growth, leaf color; too much causes soft growth and disease risk.\n- Phosphorus (P): root growth and early vigor; important early.\n- Potassium (K): fruit quality, stress tolerance, water regulation.\n\n## 3.4 Split application strategy (why it helps)\nInstead of one big fertilizer dose, split into smaller weekly doses. Benefits:\n- Less leaching, especially in sandy soils.\n- More stable growth.\n- Easier to correct if plants respond badly.\n\n## 3.5 Example program: potatoes (moderate yield target)\nAssume low organic matter and irrigated field.\n1) At planting\n   - Provide a balanced starter (includes P) near the seed piece.\n2) Early growth (2–4 weeks)\n   - Nitrogen in small weekly doses to build canopy.\n3) Tuber initiation\n   - Keep nitrogen steady but not excessive (too much delays tuber formation).\n   - Increase potassium to support tuber bulking.\n4) Bulking\n   - Prioritize potassium; monitor leaf color and avoid late nitrogen spikes.\n\nRule of thumb decision:\n- If leaves are pale from bottom up: increase N slightly.\n- If leaf edges scorch and plants look thirsty even when irrigated: check salinity and potassium.\n\n## 3.6 Micronutrients quick notes\n- Iron deficiency: yellow young leaves with green veins (common in high pH).\n- Boron: affects flowering and fruit set; both deficiency and excess are harmful.\n- Calcium: related to blossom-end rot in tomato; consistency of irrigation matters more than simply adding calcium.\n\n## 3.7 Fertigation (fertilizer through drip)\nFertigation is efficient because nutrients go where roots are. Best practices:\n- Filter water to avoid clogging.\n- Inject in the middle of the irrigation cycle, then flush with clean water.\n- Use smaller frequent doses rather than rare large doses.\n\n## 3.8 Nutrient budgeting\nTrack:\n- fertilizer type, amount, and cost\n- expected yield\n- price per kg\nThis links agronomy decisions to business decisions.', '04_postharvest_supplychain.txt': '# Chapter 4 — Post-Harvest Handling and Supply Chain Basics\n\n## 4.1 Why post-harvest matters\nFor many vegetables, post-harvest losses can be as damaging as field pests. Heat, bruising, and dehydration reduce quality and market price.\n\n## 4.2 The “3 rules”\n1) Harvest at the right maturity.\n2) Remove field heat quickly (cooling).\n3) Protect from crushing and moisture loss during transport.\n\n## 4.3 Handling lettuce (highly perishable)\nMain risks:\n- Wilting from water loss\n- Browning and decay if too warm\n- Crushing in transport\n\nBest practice workflow:\n- Harvest early morning when leaves are cool and turgid.\n- Shade immediately; avoid leaving crates in the sun.\n- Pre-cool if possible (forced-air cooling or cold room).\n- Use clean, ventilated crates; avoid overfilling.\n- Maintain airflow in the truck; do not block vents.\n- If no cold chain is available: shorten time to market, use wet burlap on crates (not soaking), and transport at night if feasible.\n\n## 4.4 Packaging and stacking\n- Strong crates prevent compression damage.\n- Stack in a stable pattern; avoid tall unstable stacks.\n- Use liners for delicate produce, but ensure ventilation.\n\n## 4.5 Simple quality grading\nSeparate produce into:\nA) premium (best price),\nB) standard (normal market),\nC) processing/animal feed (avoid mixing with A).\nThis improves overall revenue and avoids reputation damage.\n\n## 4.6 Traceability (simple)\nWrite on each batch:\n- farm name, date, crop, field section\nIf a complaint happens, you can identify which field and harvest day had issues.\n\n## 4.7 Transport checklist\n- Clean truck bed\n- Shade or cover from direct sun\n- Secure crates to avoid sliding\n- Keep produce away from chemicals and fuel odors', '05_farm_business_pricing.txt': '# Chapter 5 — Farm Business Decisions Under Price Volatility\n\n## 5.1 The reality\nWhen currency and input prices change quickly, farmers face uncertainty. A “good” agronomic decision can still be a bad financial decision if the market price drops or inputs spike.\n\n## 5.2 Build a simple cost sheet (one page)\nTrack costs per dunum or per hectare:\n- Seeds/seedlings\n- Fertilizers\n- Pesticides\n- Water and pumping\n- Labor\n- Transport and packaging\n\nThen estimate expected yield and selling price range (low / medium / high). This gives you a break-even price.\n\n## 5.3 Example: deciding whether to buy a fungicide\nAsk:\n1) How much yield is at risk without treatment?\n2) What is the probability the treatment works (timing + diagnosis)?\n3) What is the all-in cost (product + labor + fuel)?\n4) Are there cheaper alternatives (leaf removal, spacing, irrigation change)?\n\nSimple decision rule:\n- Treat only if (expected saved revenue) > (cost of treatment) and resistance risk is acceptable.\nExpected saved revenue ≈ (yield at risk) × (market price) × (probability of success).\n\n## 5.4 Managing exchange-rate risk (practical habits)\n- Keep records in both LBP and USD equivalent at purchase date.\n- If possible, buy high-impact inputs early for the season when prices are favorable.\n- Use group purchasing (co-ops) to reduce unit cost and improve negotiating power.\n\n## 5.5 Pricing your produce\nIf your costs are volatile, consider:\n- Pre-agreeing on price ranges with buyers.\n- Diversifying crops so one market crash doesn’t wipe you out.\n- Adding value (washing, grading, packaging) when it increases price more than it increases cost.\n\n## 5.6 Working with NGOs and support programs\nSome programs require evidence: field records, photos, receipts, and yield estimates. Good documentation increases your chances of support.\n\n## 5.7 Summary\nGood farming is both biology and economics. Use data to reduce uncertainty.'}

# how the model should behave
SYSTEM_RULES = """You are a retrieval-augmented digital agronomist.
Answer using ONLY the provided CONTEXT.
If the context is missing the answer, say: "I don't know based on the provided documents."
Cite chunk ids you used in [brackets] like [doc::chunk3].
Keep the answer practical, concise, and structured.
"""

# container for the documents i created
@dataclass
class Doc:
    text: str # actual content
    source: str # for file name

# creates the folder and writes the dummy files 
def ensure_dummy_docs(data_dir: str) -> None:
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, text in DUMMY_DOCS.items():
        out = path / name
        if not out.exists():
            out.write_text(text, encoding="utf-8")

# normally use RecursiveCharacterTextSplitter, but if that import fails, this function still lets chunking happen
def fallback_chunk(text: str, chunk_size: int, overlap: int) -> List[str]:
    paras = text.split("\n\n")
    chunks: List[str] = []
    cur = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(cur) + len(p) + 2 <= chunk_size:
            cur = (cur + "\n\n" + p).strip()
            continue
        if cur:
            chunks.append(cur)
        cur = p
        while len(cur) > chunk_size:
            chunks.append(cur[:chunk_size])
            cur = cur[chunk_size - overlap :]
    if cur:
        chunks.append(cur)
    return chunks

class DigitalAgronomistRAG:
    # i parameterized the class so chunking, retrieval depth, document location, and model selection can be controlled either directly or through environment variables
    def __init__(
        self,
        data_dir: str | None = None,
        embed_model: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        top_k: int = 4,
        groq_model: str | None = None,
    ) -> None:
        self.data_dir = data_dir or os.getenv("DATA_DIR", "data_agro_dummy")
        self.embed_model_name = embed_model or os.getenv(
            "ST_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunk_size = int(os.getenv("CHUNK_SIZE", chunk_size))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", chunk_overlap))
        self.top_k = int(os.getenv("TOP_K", top_k))
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.docs: List[Doc] = []
        self.chunk_texts: List[str] = []
        self.metas: List[Dict[str, str]] = []
        self.embedder: SentenceTransformer | None = None
        self.index: faiss.Index | None = None

    # loading the knowledge base from plain text files
    def load_txt_docs(self) -> List[Doc]:
        ensure_dummy_docs(self.data_dir)
        docs: List[Doc] = []
        for p in sorted(Path(self.data_dir).glob("*.txt")):
            docs.append(Doc(text=p.read_text(encoding="utf-8"), source=p.name))
        if not docs:
            raise FileNotFoundError(
                f"No .txt files found in {self.data_dir}. "
                "Either place your agronomy docs there or let ensure_dummy_docs create the demo set."
            )
        return docs

    # chunking stage
    def chunk_docs(self, docs: List[Doc]) -> Tuple[List[str], List[Dict[str, str]]]:
        texts: List[str] = []
        metas: List[Dict[str, str]] = []
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
            )
            for d in docs:
                chunks = splitter.split_text(d.text)
                for i, ch in enumerate(chunks):
                    texts.append(ch)
                    metas.append({"source": d.source, "chunk_id": f"{d.source}::chunk{i}"})
        except Exception:
            for d in docs:
                chunks = fallback_chunk(d.text, self.chunk_size, self.chunk_overlap)
                for i, ch in enumerate(chunks):
                    texts.append(ch)
                    metas.append({"source": d.source, "chunk_id": f"{d.source}::chunk{i}"})
        return texts, metas

    # indexing stage
    def build(self) -> "DigitalAgronomistRAG":
        # load and chunk docs
        self.docs = self.load_txt_docs()
        self.chunk_texts, self.metas = self.chunk_docs(self.docs)
        # embed all chunks
        self.embedder = SentenceTransformer(self.embed_model_name)
        vecs = self.embedder.encode(
            self.chunk_texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True, # the code uses IndexFlatIP, which is inner-product search so if embeddings are normalized, inner product behaves like cosine similarity
        ).astype(np.float32)
        dim = vecs.shape[1]
        # create and fill faiss index
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)
        return self
    
    # retrieval step
    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        if self.embedder is None or self.index is None:
            raise RuntimeError("Call build() before retrieve().")
        actual_top_k = top_k or self.top_k
        qvec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32) # embeds user query
        scores, idxs = self.index.search(qvec, actual_top_k) # searches faiss index
        results: List[Dict[str, Any]] = [] # top matching chunks
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "score": float(score),
                    "chunk_id": self.metas[idx]["chunk_id"],
                    "source": self.metas[idx]["source"],
                    "text": self.chunk_texts[idx],
                }
            )
        return results

    # formats the retrieved chunks into a prompt for the LLM
    def build_prompt(self, question: str, retrieved: List[Dict[str, Any]]) -> str:
        blocks = []
        for r in retrieved:
            blocks.append(f"[{r['chunk_id']}] (source={r['source']})\n{r['text']}")
        context = "\n\n".join(blocks)
        return f"""{SYSTEM_RULES}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    # sends the prompt to groq
    def call_llm(self, prompt: str) -> str:
        if not os.getenv("GROQ_API_KEY"):
            return self.fallback_answer_from_context([]) # fallback to context
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=self.groq_model,
            temperature=0.2,
            max_tokens=600,
        )
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def fallback_answer_from_context(self, retrieved: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        used = set()
        for r in retrieved:
            for ln in r["text"].splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                if ln.startswith(("-", "1)", "2)", "3)", "4)", "5)")) or "Best practice" in ln or "Decision" in ln:
                    key = (ln[:120], r["chunk_id"])
                    if key in used:
                        continue
                    used.add(key)
                    lines.append(f"- {ln} [{r['chunk_id']}]")
                if len(lines) >= 8:
                    break
            if len(lines) >= 8:
                break
        if not lines:
            lines = ["- No clear bullet lines found in the retrieved context."]
        return "Fallback answer based on retrieved context:\n" + "\n".join(lines)

    # main public interface
    def answer(self, question: str, return_metadata: bool = False) -> Dict[str, Any] | str:
        retrieved = self.retrieve(question, self.top_k) # relevant chunks
        prompt = self.build_prompt(question, retrieved)
        if os.getenv("GROQ_API_KEY"):
            answer = self.call_llm(prompt)
        else:
            answer = self.fallback_answer_from_context(retrieved)
        payload = {
            "question": question,
            "retrieved": retrieved,
            "answer": answer,
        }
        return payload if return_metadata else answer

# this wraps the RAG engine as a langchain tool named agronomy_rag_search
def make_rag_tool(rag: DigitalAgronomistRAG):
    from langchain_core.tools import tool
    @tool("agronomy_rag_search")
    def agronomy_rag_search(question: str) -> str:
        """Answer agronomy questions using the indexed farming documents only."""
        result = rag.answer(question, return_metadata=True)
        used = ", ".join(item["chunk_id"] for item in result["retrieved"])
        return (
            f"Answer:\n{result['answer']}\n\n"
            f"Retrieved chunk ids: {used}"
        )
    return agronomy_rag_search

if __name__ == "__main__":
    rag = DigitalAgronomistRAG().build()
    print(rag.answer("My zucchini leaves have white powdery spots. What should I do first?"))