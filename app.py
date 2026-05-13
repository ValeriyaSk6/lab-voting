# ІОД РІС — Лабораторні роботи №1-4
# ============================================================
import streamlit as st
import json, os, hashlib, itertools, time, random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Google Sheets storage (optional) ───────────────────────
def _get_sheet():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_dict = st.secrets.get("gcp_service_account", None)
        if creds_dict is None: return None
        scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
        client = gspread.authorize(creds)
        sheet_id = st.secrets.get("sheet_id", None)
        if sheet_id is None: return None
        sh = client.open_by_key(sheet_id)
        try: ws = sh.worksheet("data")
        except: ws = sh.add_worksheet(title="data", rows=10, cols=2); ws.append_row(["key","value"])
        return ws
    except: return None

def load_data():
    ws = _get_sheet()
    empty = {"lab1_votes":{},"lab2_heuristic_votes":{},"lab2_final_objects":[],
             "ga_result":[],"lab3_compromise":[],"lab4_satisfaction":{}}
    if ws is not None:
        try:
            for row in ws.get_all_records():
                if row.get("key") == "main": return json.loads(row["value"])
        except: pass
        return empty
    if os.path.exists("data.json"):
        with open("data.json","r",encoding="utf-8") as f: return json.load(f)
    return empty

def save_data(data):
    ws = _get_sheet(); s = json.dumps(data, ensure_ascii=False)
    if ws is not None:
        try:
            for i,row in enumerate(ws.get_all_records(), start=2):
                if row.get("key") == "main": ws.update(f"B{i}", [[s]]); return
            ws.append_row(["main", s]); return
        except: pass
    with open("data.json","w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── Config ──────────────────────────────────────────────────
OBJECTS = ["Україна","США","Велика Британія","Польща","Франція","Німеччина",
           "Канада","Японія","Австралія","Нідерланди","Швеція","Норвегія",
           "Данія","Фінляндія","Швейцарія","Чехія","Австрія","Іспанія","Португалія","Естонія"]
HEURISTICS = {
    "Е1":"Участь лише в одному МП на 3-му місці",
    "Е2":"Участь лише в одному МП на 2-му місці",
    "Е3":"Участь лише в одному МП на 1-му місці",
    "Е4":"Участь лише у 2-х МП на 3-му місці",
    "Е5":"Участь в одному МП на 3-му та одному МП на 2-му місці",
    "Е6":"Загальна кількість згадувань < 2  (авторська)",
    "Е7":"Жодного разу не отримала 1-е місце  (авторська)",
}
ADMIN_HASH = hashlib.sha256("admin2024".encode()).hexdigest()
FLAG = {"Україна":"🇺🇦","США":"🇺🇸","Велика Британія":"🇬🇧","Польща":"🇵🇱","Франція":"🇫🇷",
        "Німеччина":"🇩🇪","Канада":"🇨🇦","Японія":"🇯🇵","Австралія":"🇦🇺","Нідерланди":"🇳🇱",
        "Швеція":"🇸🇪","Норвегія":"🇳🇴","Данія":"🇩🇰","Фінляндія":"🇫🇮","Швейцарія":"🇨🇭",
        "Чехія":"🇨🇿","Австрія":"🇦🇹","Іспанія":"🇪🇸","Португалія":"🇵🇹","Естонія":"🇪🇪"}

def get_token(name):
    return hashlib.sha256(f"iod_{name.strip().lower()}_2024".encode()).hexdigest()[:8]

# ── Analytics helpers ────────────────────────────────────────
def vote_counts(votes):
    c={o:0 for o in OBJECTS}
    for vd in votes.values():
        for obj in vd.get("choices",[]):
            if obj in c: c[obj]+=1
    return c

def position_counts(votes):
    pc={o:{1:0,2:0,3:0} for o in OBJECTS}
    for vd in votes.values():
        for pos,obj in enumerate(vd.get("choices",[]),1):
            if obj in pc and pos in(1,2,3): pc[obj][pos]+=1
    return pc

def score_counts(votes):
    """3 pts for 1st, 2 for 2nd, 1 for 3rd"""
    sc={o:0 for o in OBJECTS}
    for vd in votes.values():
        for pos,obj in enumerate(vd.get("choices",[]),1):
            if obj in sc: sc[obj]+=(4-pos)
    return sc

def get_nucleus(votes):
    s=set()
    for vd in votes.values(): s.update(vd.get("choices",[]))
    return sorted(s)

def apply_heuristic(objects,votes,key):
    vc=vote_counts(votes); pc=position_counts(votes); rem=[]
    for obj in objects:
        t=vc.get(obj,0); p1=pc.get(obj,{1:0})[1]; p2=pc.get(obj,{2:0})[2]; p3=pc.get(obj,{3:0})[3]
        if   key=="Е1" and t==1 and p3==1 and p1==0 and p2==0: rem.append(obj)
        elif key=="Е2" and t==1 and p2==1 and p1==0 and p3==0: rem.append(obj)
        elif key=="Е3" and t==1 and p1==1 and p2==0 and p3==0: rem.append(obj)
        elif key=="Е4" and t==2 and p3==2 and p1==0 and p2==0: rem.append(obj)
        elif key=="Е5" and p3==1 and p2==1 and p1==0: rem.append(obj)
        elif key=="Е6" and t<2: rem.append(obj)
        elif key=="Е7" and p1==0: rem.append(obj)
    return [o for o in objects if o not in rem], rem

# ── LR3: Cook distance ───────────────────────────────────────
def cook_dist(perm, choices, n_penalty):
    """Cook distance (Е1/Е2): sum |rank_in_perm(choice_k) - k| for k=1,2,3"""
    rank = {obj:i+1 for i,obj in enumerate(perm)}
    d = 0
    for k,obj in enumerate(choices,1):
        if obj in rank: d+=abs(rank[obj]-k)
        else: d+=n_penalty
    return d

def pref_matrix(objects, votes):
    """How many experts prefer obj_i over obj_j"""
    n=len(objects); idx={o:i for i,o in enumerate(objects)}
    M=np.zeros((n,n),dtype=int)
    for vd in votes.values():
        ch=[c for c in vd.get("choices",[]) if c in idx]
        for a in range(len(ch)):
            for b in range(a+1,len(ch)):
                i,j=idx[ch[a]],idx[ch[b]]
                M[i][j]+=1
    return M

def brute_force_ranking(objects, votes):
    """Enumerate all n! permutations, return sorted by sum_dist"""
    n=len(objects); penalty=max(1,n-3)
    results=[]
    for perm in itertools.permutations(objects):
        dists=[cook_dist(list(perm), vd.get("choices",[]), penalty)
               for vd in votes.values()]
        s=sum(dists); mx=max(dists) if dists else 0
        results.append({"perm":list(perm),"sum":s,"max":mx})
    results.sort(key=lambda x:(x["sum"],x["max"]))
    return results

# ── LR2: Genetic Algorithm ───────────────────────────────────
def fitness(perm,votes):
    score=0.0
    for vd in votes.values():
        ch=[c for c in vd.get("choices",[]) if c in perm]
        for i in range(len(ch)):
            for j in range(i+1,len(ch)):
                if perm.index(ch[i])<perm.index(ch[j]): score+=3-i
    return score

def ox_crossover(p1,p2):
    n=len(p1)
    if n<2: return p1[:]
    a,b=sorted(random.sample(range(n),2))
    child=[None]*n; child[a:b+1]=p1[a:b+1]
    rem=[x for x in p2 if x not in child]; j=0
    for i in range(n):
        if child[i] is None: child[i]=rem[j]; j+=1
    return child

def genetic_algorithm(objects,votes,pop_size=80,generations=200,mut_rate=0.1):
    n=len(objects)
    if n==0: return [],[]
    pop=[random.sample(objects,n) for _ in range(pop_size)]
    best=pop[0][:]; best_f=fitness(best,votes); history=[]
    for _ in range(generations):
        scored=sorted([(ind,fitness(ind,votes)) for ind in pop],key=lambda x:x[1],reverse=True)
        if scored[0][1]>best_f: best_f=scored[0][1]; best=scored[0][0][:]
        history.append(scored[0][1])
        survivors=[ind for ind,_ in scored[:pop_size//2]]
        new_pop=survivors[:]
        while len(new_pop)<pop_size:
            p1,p2=random.sample(survivors,2); new_pop.append(ox_crossover(p1,p2))
        for i in range(1,len(new_pop)):
            if random.random()<mut_rate:
                i1,i2=random.sample(range(n),2); new_pop[i][i1],new_pop[i][i2]=new_pop[i][i2],new_pop[i][i1]
        pop=new_pop
    return best,history

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(page_title="ІОД РІС — ЛР №1-4",page_icon="🌍",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""
<style>
.main-title{background:linear-gradient(135deg,#1a3a6b 0%,#0d6e3f 100%);color:white;
  padding:20px 28px;border-radius:12px;margin-bottom:18px;text-align:center;}
.main-title h1{margin:0;font-size:1.75rem;}.main-title p{margin:4px 0 0;opacity:.85;}
.info-box{background:#e8f4fd;border-left:4px solid #1a73e8;padding:10px 14px;border-radius:6px;margin:8px 0;}
.ok-box{background:#e8f8f0;border-left:4px solid #0d6e3f;padding:10px 14px;border-radius:6px;margin:8px 0;}
.stTabs [data-baseweb="tab"]{font-size:14px;font-weight:500;}
</style>""",unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data=load_data()

st.markdown("""<div class="main-title">
<h1>🌍 Інтелектуальна обробка даних в РІС</h1>
<p>Предметна область: Пріоритетні країни для академічної мобільності студентів</p>
</div>""",unsafe_allow_html=True)

tabs=st.tabs(["🏠 Про роботу","🗳️ Голосування ЛР1","📊 Результати ЛР1",
              "📋 Евристики ЛР2","⚙️ Застосування евристик","🧬 Ген.алгоритм ЛР2",
              "📐 Колективне ранжування ЛР3","😊 Задоволеність ЛР4","🔐 Адмін"])

# ═══ TAB 0: ABOUT ═══════════════════════════════════════════
with tabs[0]:
    st.header("Про систему")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("📚 ЛР №1 — Розподілене введення даних")
        st.markdown("""**Мета:** преференційне голосування (ТОП-3 з 20 країн)
- Анонімне: ім'я → токен (SHA-256)
- Розподілене: через веб-браузер
- Результат: **ядро лідерів A^λ**, таблиця балів (3-2-1)""")
        st.subheader("📋 20 країн-об'єктів")
        for i,o in enumerate(OBJECTS,1): st.write(f"{i:>2}. {FLAG.get(o,'')} {o}")
    with c2:
        st.subheader("🔬 ЛР №2 — Розподілена обробка")
        st.markdown("""**Мета:** звуження до ≤10 об'єктів
- Відкрите голосування за евристики (2-3 з 7)
- Застосування евристик за пріоритетом
- Генетичний алгоритм (OX-crossover)""")
        st.subheader("📐 ЛР №3 — Колективне ранжування")
        st.markdown("""**Мета:** обчислення компромісного ранжування
- Матриця переваг між парами об'єктів
- Прямий перебір n! перестановок
- Метрика Кука: Е1 (поміркована взаємність) і Е2 (макс. задоволення)
- Медіанні перестановки (мінімум суми і максимуму відстаней)""")
        st.subheader("😊 ЛР №4 — Індекси задоволеності")
        st.markdown("""**Мета:** рівень задоволеності кожного експерта
- s^j = (1 − d^j / d_max) · 100%
- d^j = відстань від МП експерта до компромісного ранжування
- Симуляція розподілених обчислень""")
        st.subheader("⚙️ Евристики E1–E7")
        for k,v in HEURISTICS.items(): st.write(f"**{k}:** {v}")

# ═══ TAB 1: VOTING LR1 ══════════════════════════════════════
with tabs[1]:
    st.header("🗳️ Голосування — Лабораторна робота №1")
    st.markdown('<div class="info-box">Введіть ім\'я — воно хешується у анонімний токен. Оберіть ТОП-3 країни за пріоритетом академічної мобільності.</div>',unsafe_allow_html=True)
    name1=st.text_input("✏️ Ваше ім'я та прізвище:",placeholder="Наприклад: Марія Коваленко",key="name1")
    if not name1.strip():
        st.info("👆 Введіть ім'я, щоб продовжити.")
    else:
        expert=name1.strip(); token=get_token(expert)
        st.caption(f"Ваш анонімний токен: `{token}`")
        DATA=load_data(); votes=DATA["lab1_votes"]
        if token in votes:
            ch=votes[token]["choices"]
            st.success("✅ Ви вже проголосували!")
            st.write(f"🥇 {ch[0]}  •  🥈 {ch[1]}  •  🥉 {ch[2]}")
            if st.button("🔄 Змінити голос"):
                del DATA["lab1_votes"][token]; save_data(DATA); st.session_state.data=DATA; st.rerun()
        else:
            v1=st.selectbox("🥇 1-е місце (3 бали):",["— Оберіть —"]+OBJECTS,key="v1")
            opts2=[o for o in OBJECTS if o!=v1]
            v2=st.selectbox("🥈 2-е місце (2 бали):",["— Оберіть —"]+opts2,key="v2")
            opts3=[o for o in opts2 if o!=v2]
            v3=st.selectbox("🥉 3-є місце (1 бал):",["— Оберіть —"]+opts3,key="v3")
            if st.button("✅ Підтвердити голос",type="primary"):
                if "— Оберіть —" in[v1,v2,v3]: st.error("Оберіть усі 3 позиції!")
                elif len({v1,v2,v3})<3: st.error("Усі три країни мають бути різними!")
                else:
                    DATA["lab1_votes"][token]={"expert":expert,"choices":[v1,v2,v3],"timestamp":datetime.now().isoformat()}
                    save_data(DATA); st.session_state.data=DATA; st.balloons()
                    st.success(f"✅ Збережено! Токен: `{token}`"); st.rerun()

# ═══ TAB 2: RESULTS LR1 ═════════════════════════════════════
with tabs[2]:
    st.header("📊 Результати — Лабораторна робота №1")
    DATA=load_data(); st.session_state.data=DATA; votes=DATA["lab1_votes"]
    nucleus=get_nucleus(votes) if votes else []
    m1,m2,m3=st.columns(3)
    m1.metric("Проголосувало",len(votes))
    m2.metric("Країн в ядрі A^λ",len(nucleus))
    m3.metric("Всього об'єктів",len(OBJECTS))

    if not votes:
        st.info("Поки немає голосів.")
    else:
        vc=vote_counts(votes); pc=position_counts(votes); sc=score_counts(votes)

        # ── Ballots ──
        st.subheader("📋 Анонімні бюлетені (токени)")
        rows=[{"Токен":tkn,"🥇 1-е місце":vd["choices"][0],"🥈 2-е місце":vd["choices"][1],
               "🥉 3-є місце":vd["choices"][2],"Час":vd.get("timestamp","")[:16]}
              for tkn,vd in votes.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True)

        # ── Score table ──
        st.subheader("🏆 Таблиця балів (1-е=3б, 2-е=2б, 3-є=1б)")
        tbl=[]
        for obj in OBJECTS:
            if vc.get(obj,0)>0:
                tbl.append({"Країна":f"{FLAG.get(obj,'')} {obj}",
                             "🥇 1-е(×3)":pc[obj][1],"🥈 2-е(×2)":pc[obj][2],
                             "🥉 3-є(×1)":pc[obj][3],"Всього голосів":vc[obj],"🏅 Балів":sc[obj]})
        df=pd.DataFrame(tbl).sort_values("🏅 Балів",ascending=False).reset_index(drop=True)
        st.dataframe(df,use_container_width=True)

        # ── Chart ──
        st.subheader("📈 Діаграма балів")
        objs_sorted=df["Країна"].apply(lambda x:x.split(" ",1)[-1]).tolist()
        pts=[sc.get(o,0) for o in objs_sorted]
        colors_bar=["#FFD700" if i==0 else "#C0C0C0" if i==1 else "#CD7F32" if i==2 else "#4DABF7" for i in range(len(objs_sorted))]
        fig,ax=plt.subplots(figsize=(13,5))
        ax.bar(range(len(objs_sorted)),pts,color=colors_bar)
        ax.set_xticks(range(len(objs_sorted)))
        ax.set_xticklabels(objs_sorted,rotation=42,ha="right",fontsize=10)
        ax.set_ylabel("Бали"); ax.set_title("Рейтинг країн за системою балів (ЛР1)")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for i,(o,p) in enumerate(zip(objs_sorted,pts)): ax.text(i,p+0.1,str(p),ha="center",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Stacked bar ──
        st.subheader("📊 Розподіл місць по країнах")
        p1v=[pc[o][1] for o in objs_sorted]; p2v=[pc[o][2] for o in objs_sorted]; p3v=[pc[o][3] for o in objs_sorted]
        x=np.arange(len(objs_sorted)); fig2,ax2=plt.subplots(figsize=(13,5))
        ax2.bar(x,p1v,.65,label="🥇 1-е(3б)",color="#FFD700")
        ax2.bar(x,p2v,.65,bottom=p1v,label="🥈 2-е(2б)",color="#A8A8A8")
        ax2.bar(x,p3v,.65,bottom=[a+b for a,b in zip(p1v,p2v)],label="🥉 3-є(1б)",color="#CD7F32")
        ax2.set_xticks(x); ax2.set_xticklabels(objs_sorted,rotation=42,ha="right",fontsize=10)
        ax2.set_ylabel("Голосів"); ax2.set_title("Розподіл голосів за місцями (ЛР1)"); ax2.legend()
        ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout(); st.pyplot(fig2); plt.close()

        # ── Nucleus ──
        st.subheader(f"🎯 Ядро лідерів A^λ — {len(nucleus)} країн")
        st.markdown('<div class="ok-box">Об\'єднання підмножин МП всіх експертів (формула 2). Лише ці країни беруть участь у подальшому аналізі.</div>',unsafe_allow_html=True)
        cols=st.columns(5)
        for i,obj in enumerate(sorted(nucleus)): cols[i%5].success(f"{FLAG.get(obj,'')} {obj}")
        if not DATA.get("lab2_final_objects"):
            DATA["lab2_final_objects"]=nucleus; save_data(DATA); st.session_state.data=DATA

# ═══ TAB 3: HEURISTICS VOTING LR2 ═══════════════════════════
with tabs[3]:
    st.header("📋 Голосування за евристики — Лабораторна робота №2")
    st.markdown('<div class="info-box">Відкрите голосування. Оберіть <strong>2–3 евристики</strong>, які слід застосувати першими для відсіювання найменш підтриманих об\'єктів.</div>',unsafe_allow_html=True)
    DATA=load_data(); st.session_state.data=DATA
    name2=st.text_input("✏️ Ваше ім'я та прізвище:",placeholder="Наприклад: Марія Коваленко",key="name2")
    if not name2.strip():
        st.info("👆 Введіть ім'я, щоб продовжити.")
    else:
        exp2=name2.strip()
        if exp2 in DATA["lab2_heuristic_votes"]:
            st.success(f"✅ Ви вже обрали: **{', '.join(DATA['lab2_heuristic_votes'][exp2])}**")
            if st.button("🔄 Змінити вибір"):
                del DATA["lab2_heuristic_votes"][exp2]; save_data(DATA); st.session_state.data=DATA; st.rerun()
        else:
            sel=[k for k,v in HEURISTICS.items() if st.checkbox(f"**{k}** — {v}",key=f"h_{k}")]
            st.caption(f"Обрано: {len(sel)}")
            if st.button("✅ Зберегти вибір",type="primary"):
                if not(2<=len(sel)<=3): st.error("Оберіть від 2 до 3 евристик!")
                else:
                    DATA["lab2_heuristic_votes"][exp2]=sel; save_data(DATA); st.session_state.data=DATA; st.success("✅ Збережено!"); st.rerun()
    if DATA["lab2_heuristic_votes"]:
        st.markdown("---"); st.subheader("📊 Підрахунок голосів за евристиками")
        hc={k:0 for k in HEURISTICS}
        for hv in DATA["lab2_heuristic_votes"].values():
            for h in hv:
                if h in hc: hc[h]+=1
        hdf=pd.DataFrame([{"Евристика":k,"Опис":HEURISTICS[k],"Голосів":hc[k]} for k in HEURISTICS]).sort_values("Голосів",ascending=False).reset_index(drop=True)
        st.dataframe(hdf,use_container_width=True)
        fig3,ax3=plt.subplots(figsize=(10,4))
        ax3.barh(hdf["Евристика"][::-1],hdf["Голосів"][::-1],color="#1a73e8")
        ax3.set_xlabel("Голосів"); ax3.set_title("Пріоритетність евристик")
        ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True)); plt.tight_layout(); st.pyplot(fig3); plt.close()
        st.subheader("📄 Протокол голосування за евристиками")
        st.dataframe(pd.DataFrame([{"Учасник":e,"Обрані евристики":", ".join(hv)} for e,hv in DATA["lab2_heuristic_votes"].items()]),use_container_width=True)

# ═══ TAB 4: APPLY HEURISTICS LR2 ════════════════════════════
with tabs[4]:
    st.header("⚙️ Застосування евристик — Лабораторна робота №2")
    DATA=load_data(); st.session_state.data=DATA; votes=DATA["lab1_votes"]; nucleus=get_nucleus(votes) if votes else []
    if not nucleus:
        st.warning("Спочатку проведіть голосування ЛР1.")
    else:
        st.markdown(f'<div class="info-box">Початкове ядро A^λ: <strong>{len(nucleus)} країн</strong> — {", ".join(nucleus)}</div>',unsafe_allow_html=True)
        hc={k:0 for k in HEURISTICS}
        for hv in DATA["lab2_heuristic_votes"].values():
            for h in hv:
                if h in hc: hc[h]+=1
        priority=sorted([k for k in HEURISTICS if hc[k]>0],key=lambda k:hc[k],reverse=True) or list(HEURISTICS.keys())
        st.subheader("🔢 Порядок застосування евристик (за голосами)")
        for i,h in enumerate(priority,1): st.write(f"{i}. **{h}** — {HEURISTICS[h]}  *({hc[h]} голос.)*")
        st.markdown("---"); st.subheader("🔄 Покрокове відсіювання")
        current=nucleus[:]; steps=[{"Крок":"Початкове ядро","n":len(current),"Країни":", ".join(current),"Видалено":"—"}]
        for h in priority:
            if len(current)<=10: break
            new_set,removed=apply_heuristic(current,votes,h)
            if removed:
                current=new_set; steps.append({"Крок":f"Після {h}","n":len(current),"Країни":", ".join(current),"Видалено":", ".join(removed)})
        for step in steps:
            icon="🟢" if step["n"]<=10 else "🟡"
            with st.expander(f"{icon} {step['Крок']}: залишилось **{step['n']}** країн",expanded=True):
                if step["Видалено"]!="—": st.error(f"❌ Видалено: {step['Видалено']}")
                st.success(f"✅ Залишилось: {step['Країни']}")
        st.subheader("📉 Графік зменшення підмножини")
        sizes=[s["n"] for s in steps]; labels=[s["Крок"] for s in steps]
        fig4,ax4=plt.subplots(figsize=(9,4))
        ax4.plot(labels,sizes,"o-",color="#1a3a6b",linewidth=2,markersize=8)
        ax4.axhline(10,color="red",linestyle="--",label="Межа ≤ 10")
        ax4.set_ylabel("Країн"); ax4.set_title("Зменшення підмножини при застосуванні евристик")
        ax4.set_xticks(range(len(labels))); ax4.set_xticklabels(labels,rotation=30,ha="right")
        ax4.legend(); ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); plt.tight_layout(); st.pyplot(fig4); plt.close()
        st.subheader(f"🏁 Фінальна підмножина: {len(current)} країн")
        for i,obj in enumerate(current,1): st.write(f"{i}. {FLAG.get(obj,'')} {obj}")
        if st.button("💾 Зберегти фінальну підмножину",type="primary"):
            DATA["lab2_final_objects"]=current; save_data(DATA); st.session_state.data=DATA; st.success("✅ Збережено!")

# ═══ TAB 5: GENETIC ALGORITHM LR2 ═══════════════════════════
with tabs[5]:
    st.header("🧬 Генетичний алгоритм — Лабораторна робота №2")
    DATA=load_data(); st.session_state.data=DATA; final_objs=DATA.get("lab2_final_objects",[])
    if not final_objs:
        st.warning("Збережіть фінальну підмножину на вкладці '⚙️ Застосування евристик'.")
    else:
        st.write(f"**Підмножина ({len(final_objs)} країн):** {', '.join(final_objs)}")
        with st.expander("📖 Опис генетичного алгоритму"):
            st.markdown("""| Оператор | Реалізація |\n|---|---|\n| Ініціалізація | Випадкові перестановки |\n| Схрещування | **Order Crossover (OX)** |\n| Мутація | Обмін двох елементів з ймовірністю p |\n| Селекція | Елітний відбір 50% найкращих |\n| Пристосованість | Кількість правильно впорядкованих пар МП |""")
        c1,c2,c3=st.columns(3)
        pop_size=c1.slider("Популяція",20,300,80,10)
        generations=c2.slider("Поколінь",20,1000,200,20)
        mut_rate=c3.slider("Мутація p",0.01,0.50,0.10,0.01)
        if st.button("▶️ Запустити ГА",type="primary"):
            with st.spinner("⏳ Виконується..."):
                result,history=genetic_algorithm(final_objs,DATA["lab1_votes"],pop_size,generations,mut_rate)
            DATA["ga_result"]=result; save_data(DATA); st.session_state.data=DATA; st.success("✅ Готово!")
            st.subheader("🏆 Результат ГА:")
            medals=["🥇","🥈","🥉"]
            for i,obj in enumerate(result,1):
                st.markdown(f"{medals[i-1] if i<=3 else str(i)+'.'} {FLAG.get(obj,'')} {obj}")
            fig5,ax5=plt.subplots(figsize=(10,4)); ax5.plot(history,color="#0d6e3f",linewidth=1.5)
            ax5.set_xlabel("Покоління"); ax5.set_ylabel("Пристосованість"); ax5.set_title("Збіжність ГА"); plt.tight_layout(); st.pyplot(fig5); plt.close()
        elif DATA.get("ga_result"):
            st.subheader("🏆 Попередній результат ГА:")
            medals=["🥇","🥈","🥉"]
            for i,obj in enumerate(DATA["ga_result"],1):
                st.markdown(f"{medals[i-1] if i<=3 else str(i)+'.'} {FLAG.get(obj,'')} {obj}")
            st.info("Натисніть '▶️ Запустити ГА' для оновлення.")

# ═══ TAB 6: COLLECTIVE RANKING LR3 ══════════════════════════
with tabs[6]:
    st.header("📐 Колективне ранжування — Лабораторна робота №3")
    DATA=load_data(); st.session_state.data=DATA
    votes=DATA["lab1_votes"]; final_objs=DATA.get("lab2_final_objects",[])
    if not final_objs or not votes:
        st.warning("Спочатку виконайте ЛР1 та ЛР2 (збережіть фінальну підмножину).")
    else:
        n=len(final_objs)
        st.markdown(f'<div class="info-box"><strong>Фінальна підмножина:</strong> {n} країн — {", ".join(final_objs)}<br>Буде перебрано <strong>{n}! = {__import__("math").factorial(n):,}</strong> перестановок.</div>',unsafe_allow_html=True)

        # ── 1. Expert MPs table ──
        st.subheader("1️⃣ Задані МП експертів")
        mp_rows=[]
        for tkn,vd in votes.items():
            ch=vd["choices"]; mp_rows.append({"Токен":tkn,"1-е місце":ch[0],"2-е місце":ch[1],"3-є місце":ch[2]})
        st.dataframe(pd.DataFrame(mp_rows),use_container_width=True)

        # ── 2. Preference matrix ──
        st.subheader("2️⃣ Матриця переваг (скільки експертів надали перевагу рядку перед стовпцем)")
        M=pref_matrix(final_objs,votes)
        mdf=pd.DataFrame(M,index=final_objs,columns=final_objs)
        st.dataframe(mdf.style.background_gradient(cmap="Blues"),use_container_width=True)

        # ── 3. Brute force ──
        st.subheader("3️⃣ Прямий перебір — метрика Кука")
        st.markdown("""**Евристика Е1 (поміркована взаємність):** d^l = Σₖ |rank_R(aₖ) − k|
**Евристика Е2 (максимальне задоволення):** те ж саме обчислення для кожної пари з МП""")

        if n<=8:
            if st.button("▶️ Запустити прямий перебір",type="primary"):
                t0=time.time()
                with st.spinner(f"⏳ Перебираємо {__import__('math').factorial(n):,} перестановок..."):
                    results=brute_force_ranking(final_objs,votes)
                elapsed=time.time()-t0
                DATA["lab3_compromise"]=[results[0]["perm"]]
                save_data(DATA); st.session_state.data=DATA
                st.success(f"✅ Готово за {elapsed:.2f} с! Знайдено {len([r for r in results if r['sum']==results[0]['sum']])} оптимальних ранжувань.")

                # Show first 10
                st.subheader("Перші 10 перестановок (відсортовано за сумою відстаней)")
                disp=[]
                for r in results[:10]:
                    disp.append({"Ранжування":" > ".join([f"{FLAG.get(o,'')} {o}" for o in r["perm"]]),
                                  "Сума відстаней (E1)":r["sum"],"Макс відстань (E2)":r["max"]})
                st.dataframe(pd.DataFrame(disp),use_container_width=True)

                # Best result
                best=results[0]
                st.subheader("🏆 Компромісне ранжування (медіана):")
                medals=["🥇","🥈","🥉"]
                for i,obj in enumerate(best["perm"],1):
                    st.markdown(f"{medals[i-1] if i<=3 else str(i)+'.'} {FLAG.get(obj,'')} {obj}")
                st.info(f"Мінімальна сума відстаней: **{best['sum']}**, максимум: **{best['max']}**")

                # Chart: distribution of sum distances
                sums=[r["sum"] for r in results]
                fig6,ax6=plt.subplots(figsize=(10,4))
                ax6.hist(sums,bins=30,color="#1a3a6b",edgecolor="white")
                ax6.axvline(best["sum"],color="red",linestyle="--",label=f"Мінімум: {best['sum']}")
                ax6.set_xlabel("Сума відстаней Кука"); ax6.set_ylabel("Кількість перестановок")
                ax6.set_title(f"Розподіл суми відстаней для всіх {__import__('math').factorial(n):,} перестановок"); ax6.legend()
                plt.tight_layout(); st.pyplot(fig6); plt.close()

            elif DATA.get("lab3_compromise"):
                best_perm=DATA["lab3_compromise"][0]
                st.subheader("🏆 Збережене компромісне ранжування:")
                medals=["🥇","🥈","🥉"]
                for i,obj in enumerate(best_perm,1):
                    st.markdown(f"{medals[i-1] if i<=3 else str(i)+'.'} {FLAG.get(obj,'')} {obj}")
                st.info("Натисніть '▶️ Запустити прямий перебір' для повного перерахунку.")
            else:
                st.info("Натисніть '▶️ Запустити прямий перебір' для обчислення.")
        else:
            st.warning(f"n={n} > 8. Прямий перебір надто великий. Використовуйте ГА (вкладка ЛР2).")

        # ── 4. GA comparison ──
        if DATA.get("ga_result") and DATA.get("lab3_compromise"):
            st.subheader("4️⃣ Порівняння: прямий перебір vs ГА")
            bf=DATA["lab3_compromise"][0]; ga=DATA["ga_result"]
            penalty=max(1,n-3)
            dist_bf=sum(cook_dist(bf,vd.get("choices",[]),penalty) for vd in votes.values())
            dist_ga=sum(cook_dist(ga,vd.get("choices",[]),penalty) for vd in votes.values())
            comp=pd.DataFrame([{"Метод":"Прямий перебір","Ранжування":" > ".join(bf[:4])+(" ..." if len(bf)>4 else ""),"Сума відстаней Кука":dist_bf},
                                {"Метод":"Генетичний алгоритм","Ранжування":" > ".join(ga[:4])+(" ..." if len(ga)>4 else ""),"Сума відстаней Кука":dist_ga}])
            st.dataframe(comp,use_container_width=True)

# ═══ TAB 7: SATISFACTION LR4 ════════════════════════════════
with tabs[7]:
    st.header("😊 Індекси задоволеності — Лабораторна робота №4")
    DATA=load_data(); st.session_state.data=DATA
    votes=DATA["lab1_votes"]; compromise=DATA.get("lab3_compromise",[])
    final_objs=DATA.get("lab2_final_objects",[])

    if not compromise or not votes or not final_objs:
        st.warning("Спочатку виконайте ЛР3 (прямий перебір) і отримайте компромісне ранжування.")
    else:
        best_perm=compromise[0]; n=len(best_perm)
        rank_in_best={obj:i+1 for i,obj in enumerate(best_perm)}
        d_max=(n-1)+(n-2)+(n-3)  # max possible distance for 3 objects in ranking of n

        st.markdown(f'<div class="info-box"><strong>Компромісне ранжування R*:</strong> {" ≻ ".join([FLAG.get(o,"")+o for o in best_perm])}</div>',unsafe_allow_html=True)
        st.markdown(f"**Формула:** s^j = (1 − d^j / d_max) · 100%,  де d_max = {d_max}")

        # ── Compute satisfaction ──
        sat_rows=[]
        for tkn,vd in votes.items():
            ch=vd.get("choices",[]); expert=vd.get("expert","?")
            d=0
            for k,obj in enumerate(ch,1):
                if obj in rank_in_best: d+=abs(rank_in_best[obj]-k)
                else: d+=n-3  # penalty for eliminated object
            s=max(0.0, (1-d/d_max)*100) if d_max>0 else 100.0
            sat_rows.append({"Учасник":expert,"Токен":tkn,
                              "1-е місце":ch[0],"2-е місце":ch[1],"3-є місце":ch[2],
                              "Відстань d^j":d,"Задоволеність s^j, %":round(s,1)})

        sat_df=pd.DataFrame(sat_rows).sort_values("Задоволеність s^j, %",ascending=False).reset_index(drop=True)
        DATA["lab4_satisfaction"]={row["Токен"]:{"s":row["Задоволеність s^j, %"],"d":row["Відстань d^j"]} for _,row in sat_df.iterrows()}
        save_data(DATA); st.session_state.data=DATA

        st.subheader("📋 Індекси задоволеності експертів")
        st.dataframe(sat_df,use_container_width=True)

        avg_s=sat_df["Задоволеність s^j, %"].mean()
        c1,c2,c3=st.columns(3)
        c1.metric("Середня задоволеність",f"{avg_s:.1f}%")
        c2.metric("Максимальна",f"{sat_df['Задоволеність s^j, %'].max():.1f}%")
        c3.metric("Мінімальна",f"{sat_df['Задоволеність s^j, %'].min():.1f}%")

        # ── Bar chart of satisfaction ──
        st.subheader("📊 Діаграма задоволеності")
        fig7,ax7=plt.subplots(figsize=(12,5))
        colors7=["#0d6e3f" if s>=70 else "#FFD700" if s>=40 else "#c0392b" for s in sat_df["Задоволеність s^j, %"]]
        bars=ax7.bar(range(len(sat_df)),sat_df["Задоволеність s^j, %"],color=colors7)
        ax7.set_xticks(range(len(sat_df)))
        ax7.set_xticklabels([r["Учасник"].split()[-1] for _,r in sat_df.iterrows()],rotation=45,ha="right",fontsize=9)
        ax7.axhline(avg_s,color="blue",linestyle="--",label=f"Середнє: {avg_s:.1f}%")
        ax7.set_ylabel("Задоволеність, %"); ax7.set_title("Індекси задоволеності експертів компромісним ранжуванням")
        ax7.set_ylim(0,110); ax7.legend()
        for i,v in enumerate(sat_df["Задоволеність s^j, %"]): ax7.text(i,v+1,f"{v:.0f}%",ha="center",fontsize=8)
        patch_hi=mpatches.Patch(color="#0d6e3f",label="Висока (≥70%)"); patch_med=mpatches.Patch(color="#FFD700",label="Середня (40–70%)"); patch_lo=mpatches.Patch(color="#c0392b",label="Низька (<40%)")
        ax7.legend(handles=[patch_hi,patch_med,patch_lo,plt.Line2D([0],[0],color="blue",linestyle="--",label=f"Середнє {avg_s:.1f}%")],loc="upper right",fontsize=8)
        plt.tight_layout(); st.pyplot(fig7); plt.close()

        # ── Situation B: distributed simulation ──
        st.subheader("🖥️ Ситуація Б — симуляція розподілених обчислень")
        import math
        fo=final_objs; n_obj=len(fo)
        total_perms=math.factorial(n_obj)

        st.markdown(f"**Завдання:** розподілити {total_perms:,} перестановок між кількома вузлами")
        n_nodes=st.slider("Кількість вузлів",2,8,4)
        chunk=total_perms//n_nodes
        node_rows=[{"Вузол":f"Node {i+1}","Перестановок":chunk+(total_perms%n_nodes if i==n_nodes-1 else 0),
                    "Перші пермут.":str(list(fo[:3]))+"...","Час (симуляція, мс)":round(random.uniform(80,150),1)} for i in range(n_nodes)]
        st.dataframe(pd.DataFrame(node_rows),use_container_width=True)

        t_central=total_perms*0.000005
        t_distrib=t_central/n_nodes*1.15
        c1,c2,c3=st.columns(3)
        c1.metric("Централізований (мс)",f"{t_central*1000:.1f}")
        c2.metric(f"Розподілений ({n_nodes} вузли, мс)",f"{t_distrib*1000:.1f}")
        c3.metric("Прискорення",f"{t_central/t_distrib:.1f}×")

        ns_range=list(range(2,9))
        speedups=[math.factorial(n_obj)*0.000005/(math.factorial(n_obj)*0.000005/nv*1.15) for nv in ns_range]
        fig8,ax8=plt.subplots(figsize=(8,4))
        ax8.plot(ns_range,speedups,"o-",color="#1a3a6b",linewidth=2,markersize=8)
        ax8.plot(ns_range,ns_range,"--",color="gray",label="Ідеальне прискорення")
        ax8.set_xlabel("Кількість вузлів"); ax8.set_ylabel("Прискорення")
        ax8.set_title("Прискорення при розподілених обчисленнях"); ax8.legend()
        ax8.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); plt.tight_layout(); st.pyplot(fig8); plt.close()

# ═══ TAB 8: ADMIN ════════════════════════════════════════════
with tabs[8]:
    st.header("🔐 Адміністративна панель")
    st.caption("Пароль: `admin2024`")
    pwd=st.text_input("🔑 Пароль:",type="password",key="admin_pwd")
    if not pwd:
        st.info("Введіть пароль для доступу.")
    elif hashlib.sha256(pwd.encode()).hexdigest()!=ADMIN_HASH:
        st.error("❌ Невірний пароль")
    else:
        st.success("✅ Доступ надано")
        DATA=load_data(); votes=DATA["lab1_votes"]; hv=DATA["lab2_heuristic_votes"]
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Голосів ЛР1",len(votes)); m2.metric("Голосів за евристики",len(hv))
        m3.metric("Об'єктів (фінал)",len(DATA.get("lab2_final_objects",[])))
        m4.metric("Компромісних ранжувань",len(DATA.get("lab3_compromise",[])))
        st.markdown("---")
        st.subheader("📋 Повний протокол ЛР1 (з іменами)")
        if votes:
            prot=[{"Токен":t,"Ім'я":vd.get("expert","—"),"🥇":vd["choices"][0],"🥈":vd["choices"][1],"🥉":vd["choices"][2],"Час":vd.get("timestamp","")[:19]} for t,vd in votes.items()]
            st.dataframe(pd.DataFrame(prot),use_container_width=True)
        else: st.info("Немає голосів.")
        st.subheader("📋 Протокол евристик")
        if hv: st.dataframe(pd.DataFrame([{"Учасник":e,"Евристики":", ".join(h)} for e,h in hv.items()]),use_container_width=True)
        else: st.info("Немає.")
        if DATA.get("lab3_compromise"):
            st.subheader("📐 Компромісне ранжування (ЛР3)")
            st.write(" ≻ ".join([FLAG.get(o,"")+o for o in DATA["lab3_compromise"][0]]))
        if DATA.get("lab4_satisfaction"):
            st.subheader("😊 Збережені індекси задоволеності (ЛР4)")
            rows=[{"Токен":t,"s^j, %":v["s"],"d^j":v["d"]} for t,v in DATA["lab4_satisfaction"].items()]
            st.dataframe(pd.DataFrame(rows),use_container_width=True)
        st.markdown("---"); st.subheader("🗑️ Управління даними")
        c1,c2,c3=st.columns(3)
        if c1.button("🗑️ Очистити голоси ЛР1"):
            DATA["lab1_votes"]={};DATA["lab2_final_objects"]=[];DATA["lab3_compromise"]=[];DATA["lab4_satisfaction"]={}
            save_data(DATA);st.session_state.data=DATA;st.success("Очищено.");st.rerun()
        if c2.button("🗑️ Очистити евристики"):
            DATA["lab2_heuristic_votes"]={};save_data(DATA);st.session_state.data=DATA;st.success("Очищено.");st.rerun()
        if c3.button("⚠️ Скинути ВСЕ"):
            fresh={"lab1_votes":{},"lab2_heuristic_votes":{},"lab2_final_objects":[],"ga_result":[],"lab3_compromise":[],"lab4_satisfaction":{}}
            save_data(fresh);st.session_state.data=fresh;st.warning("Скинуто!");st.rerun()
        st.markdown("---"); st.subheader("📥 Завантажити дані")
        st.download_button("⬇️ data.json",data=json.dumps(DATA,ensure_ascii=False,indent=2).encode("utf-8"),file_name="voting_data.json",mime="application/json")
