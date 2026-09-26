from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sympy as sp
import re

app = FastAPI(title="حل‌گر ریاضی فارسی", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

class SolveRequest(BaseModel):
    question: str

def normalize_fa(s: str) -> str:
    table = str.maketrans({"۰":"0","۱":"1","۲":"2","۳":"3","۴":"4","۵":"5","۶":"6","۷":"7","۸":"8","۹":"9","×":"*","÷":"/","−":"-","–":"-","—":"-","٫":".","،":","})
    s = s.translate(table)
    repl = {"به توان":"^","توان":"^","ضرب":"*","در":"*","بعلاوه":"+","به علاوه":"+","منهای":"-","منهایِ":"-","تقسیم بر":"/","مساوی":"=","سینوس":"sin","کسینوس":"cos","تانژانت":"tan","تَنژانت":"tan","لگاریتم":"log","ln":"log","ریشه":"sqrt"}
    for a,b in sorted(repl.items(), key=lambda x: -len(x[0])): s = s.replace(a,b)
    s = s.replace("^","**")
    s = re.sub(r"(?<=\d)\s+(?=\()", "*", s)
    return s.strip()

def local_dict():
    x,y,z,t,a,b,c = sp.symbols("x y z t a b c")
    return {"x":x,"y":y,"z":z,"t":t,"a":a,"b":b,"c":c,"pi":sp.pi,"e":sp.E,"I":sp.I,"sin":sp.sin,"cos":sp.cos,"tan":sp.tan,"asin":sp.asin,"acos":sp.acos,"atan":sp.atan,"sqrt":sp.sqrt,"log":sp.log,"exp":sp.exp,"Abs":sp.Abs}

def parse_expr(s): return sp.sympify(normalize_fa(s), locals=local_dict())
def result(kind, answer, steps=None): return {"ok":True,"type":kind,"answer":str(answer),"latex":sp.latex(answer) if isinstance(answer, sp.Basic) else "","steps":steps or []}

def solve_question(q: str):
    raw=q.strip(); n=normalize_fa(raw); low=raw.lower()
    if re.search(r"مشتق|derivative|diff", low):
        m=re.search(r"(?:مشتق|derivative|diff)\s*(?:از|of)?\s*(.+?)(?:\s*(?:نسبت به|با توجه به|relative to)\s*([a-zA-Z]))?$", raw, re.I)
        expr_text=m.group(1) if m else re.sub(r"^(مشتق|derivative|diff)\s*", "", raw, flags=re.I); var_text=m.group(2) if m and m.group(2) else "x"
        expr=parse_expr(expr_text); var=local_dict().get(var_text, sp.Symbol(var_text)); ans=sp.diff(expr,var)
        return result("مشتق",ans,[f"تابع: {expr}",f"متغیر: {var}",f"مشتق: {ans}"])
    if re.search(r"انتگرال|integral|integrate", low):
        m=re.search(r"(?:انتگرال|integral|integrate)\s*(?:از)?\s*(.+?)(?:\s*(?:نسبت به|با توجه به|dx|d)\s*([a-zA-Z]))?$", raw, re.I)
        expr_text=m.group(1) if m else re.sub(r"^(انتگرال|integral|integrate)\s*", "", raw, flags=re.I); var_text=m.group(2) if m and m.group(2) else "x"
        expr=parse_expr(expr_text); var=local_dict().get(var_text, sp.Symbol(var_text)); ans=sp.integrate(expr,var)
        return result("انتگرال",ans,[f"تابع: {expr}",f"متغیر: {var}",f"انتگرال: {ans}"])
    if re.search(r"حد|limit", low):
        m=re.search(r"(?:حد|limit)\s*(?:تابع)?\s*(.+?)\s*(?:وقتی|در|as)?\s*([a-zA-Z])?\s*(?:به|->|→)?\s*(-?\d+(?:\.\d+)?)", raw, re.I)
        if not m: raise ValueError("برای حد، مثلاً بنویس: حد (sin(x)/x) وقتی x به صفر میل می‌کند")
        expr=parse_expr(m.group(1)); var=local_dict().get(m.group(2) or "x",sp.Symbol(m.group(2) or "x")); point=sp.sympify(m.group(3)); ans=sp.limit(expr,var,point)
        return result("حد",ans,[f"تابع: {expr}",f"{var} → {point}",f"حد: {ans}"])
    if re.search(r"تجزیه|factor", low):
        expr=parse_expr(re.sub(r"^(تجزیه|factor)\s*", "", raw, flags=re.I)); ans=sp.factor(expr); return result("تجزیه",ans,[f"عبارت: {expr}",f"تجزیه: {ans}"])
    if re.search(r"بسط|expand", low):
        expr=parse_expr(re.sub(r"^(بسط|expand)\s*", "", raw, flags=re.I)); ans=sp.expand(expr); return result("بسط",ans,[f"عبارت: {expr}",f"بسط: {ans}"])
    if re.search(r"ساده|simplify", low):
        expr=parse_expr(re.sub(r"^(ساده\s*کن|ساده|simplify)\s*", "", raw, flags=re.I)); ans=sp.simplify(expr); return result("ساده‌سازی",ans,[f"عبارت: {expr}",f"نتیجه: {ans}"])
    if "=" in n:
        parts=[p.strip() for p in n.split("=",1)]; lhs,rhs=map(parse_expr,parts); vars_found=sorted((lhs-rhs).free_symbols,key=lambda s:s.name)
        if not vars_found: return result("بررسی تساوی",sp.simplify(lhs-rhs)==0,[f"سمت چپ: {lhs}",f"سمت راست: {rhs}"])
        ans=sp.solve(sp.Eq(lhs,rhs),vars_found,dict=True)
        return {"ok":True,"type":"معادله","answer":str(ans),"latex":"","steps":[f"معادله: {lhs} = {rhs}",f"متغیرها: {', '.join(map(str,vars_found))}",f"جواب: {ans}"]}
    expr=parse_expr(n); ans=sp.simplify(expr); return result("محاسبه",ans,[f"عبارت: {expr}",f"نتیجه ساده‌شده: {ans}"])

@app.get("/api/health")
def health(): return {"ok":True,"service":"math-solver","version":"1.0.0"}

@app.post("/api/solve")
def solve(req: SolveRequest):
    try:
        if not req.question.strip(): return {"ok":False,"error":"لطفاً مسئله را وارد کنید."}
        return solve_question(req.question)
    except Exception as e:
        return {"ok":False,"error":"نتوانستم این عبارت را تفسیر کنم.","details":str(e)}

HTML = r'''<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="theme-color" content="#111827"><title>حل‌گر ریاضی فارسی</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f4f6;color:#111827;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Tahoma,sans-serif}.app{max-width:760px;margin:auto;padding:18px}header{display:flex;justify-content:space-between;align-items:center;padding:10px 4px 18px}h1{margin:0;font-size:30px}header p{margin:5px 0;color:#6b7280}header span{background:#111827;color:white;padding:9px 12px;border-radius:12px}.card{background:white;border-radius:20px;padding:18px;margin-bottom:15px;box-shadow:0 4px 18px #00000010}label,h2{font-weight:700}h2{margin:0 0 12px}textarea{width:100%;min-height:145px;border:1px solid #d1d5db;border-radius:14px;padding:14px;font-size:18px;resize:vertical;direction:rtl}.chips{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}.chips button,#copy{border:0;border-radius:12px;padding:9px 12px;background:#e5e7eb;font-size:14px}.primary{width:100%;border:0;border-radius:14px;padding:14px;background:#111827;color:white;font-size:18px;font-weight:700}.rh{display:flex;justify-content:space-between;align-items:center}.answer{white-space:pre-wrap;background:#f9fafb;border-radius:14px;padding:15px;min-height:70px;font-size:18px}.error{color:#b91c1c}li{margin:9px 0;line-height:1.6}</style></head><body><div class="app"><header><div><h1>حل‌گر ریاضی</h1><p>موتور Python + SymPy</p></div><span>FA</span></header><section class="card"><label>مسئله را وارد کنید</label><textarea id="q" placeholder="مثلاً: x^2 - 5x + 6 = 0"></textarea><div class="chips"><button data-q="x^2 - 5x + 6 = 0">معادله</button><button data-q="مشتق x^3 را حساب کن">مشتق</button><button data-q="انتگرال x^2 نسبت به x">انتگرال</button><button data-q="حد sin(x)/x وقتی x به صفر میل می کند">حد</button><button data-q="تجزیه x^2 - 5*x + 6">تجزیه</button></div><button id="go" class="primary">حل کن</button></section><section class="card"><div class="rh"><h2>نتیجه</h2><button id="copy">کپی</button></div><div id="answer" class="answer">هنوز مسئله‌ای حل نشده است.</div></section><section class="card"><h2>مراحل حل</h2><ol id="steps"><li>—</li></ol></section></div><script>
const q=document.getElementById('q'),answer=document.getElementById('answer'),steps=document.getElementById('steps');document.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>q.value=b.dataset.q);document.getElementById('go').onclick=async()=>{const text=q.value.trim();if(!text){answer.textContent='لطفاً مسئله را وارد کنید.';return}answer.textContent='در حال حل...';try{const r=await fetch('/api/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:text})});const d=await r.json();if(!d.ok){answer.textContent=d.error+(d.details?'\n'+d.details:'');return}answer.textContent=d.answer;steps.innerHTML=(d.steps||[]).map(x=>'<li>'+String(x).replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[m]))+'</li>').join('')}catch(e){answer.textContent='ارتباط با سرور برقرار نشد.'}};document.getElementById('copy').onclick=()=>navigator.clipboard?.writeText(answer.textContent);q.addEventListener('keydown',e=>{if((e.metaKey||e.ctrlKey)&&e.key==='Enter')document.getElementById('go').click()});</script></body></html>'''

@app.get("/", response_class=HTMLResponse)
def home(): return HTML
