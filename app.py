import streamlit as st
import json
import os
import hashlib
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

def _get_sheet():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        creds_dict = st.secrets.get("gcp_service_account", None)
        if creds_dict is None:
            return None
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(dict(creds_dict), scopes=scopes)
        client = gspread.authorize(creds)
        sheet_id = st.secrets.get("sheet_id", None)
        if sheet_id is None:
            return None
        sh = client.open_by_key(sheet_id)
        try:
            ws = sh.worksheet("data")
        except Exception:
            ws = sh.add_worksheet(title="data", rows=10, cols=2)
            ws.append_row(["key","value"])
        return ws
    except Exception:
        return None

def load_data():
    ws = _get_sheet()
    empty = {"lab1_votes":{},"lab2_heuristic_votes":{},"lab2_final_objects":[],"ga_result":[]}
    if ws is not None:
        try:
            for row in ws.get_all_records():
                if row.get("key") == "main":
                    return json.loads(row["value"])
        except Exception:
            pass
        return empty
    if os.path.exists("data.json"):
        with open("data.json","r",encoding="utf-8") as f:
            return json.load(f)
    return empty

def save_data(data):
    ws = _get_sheet()
    s = json.dumps(data, ensure_ascii=False)
    if ws is not None:
        try:
            for i, row in enumerate(ws.get_all_records(), start=2):
                if row.get("key") == "main":
                    ws.update(f"B{i}", [[s]]); return
            ws.append_row(["main", s]); return
        except Exception:
            pass
    with open("data.json","w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

OBJECTS = [
    "Україна","США","Велика Британія","Польща","Франція",
    "Німеччина","Канада","Японія","Австралія","Нідерланди",
    "Швеція","Норвегія","Данія","Фінляндія","Швейцарія",
    "Чехія","Австрія","Іспанія","Португалія","Естонія"
]
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
FLAG = {
    "Україна":"🇺🇦","США":"🇺🇸","Велика Британія":"🇬🇧","Польща":"🇵🇱","Франція":"🇫🇷",
    "Німеччина":"🇩🇪","Канада":"🇨🇦","Японія":"🇯🇵","Австралія":"🇦🇺","Нідерланди":"🇳🇱",
    "Швеція":"🇸🇪","Норвегія":"🇳🇴","Данія":"🇩🇰","Фінляндія":"🇫🇮","Швейцарія":"🇨🇭",
    "Чехія":"🇨🇿","Австрія":"🇦🇹","Іспанія":"🇪🇸","Португалія":"🇵🇹","Естонія":"🇪🇪",
}

def get_token(name):
    return hashlib.sha256(f"iod_{name.strip().lower()}_2024".encode()).hexdigest()[:8]

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

st.set_page_config(page_title="ІОД РІС — ЛР №1 і №2",page_icon="🌍",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""
<style>
.main-title{background:linear-gradient(135deg,#1a3a6b 0%,#0d6e3f 100%);color:white;padding:22px 28px;border-radius:12px;margin-bottom:20px;text-align:center;}
.main-title h1{margin:0;font-size:1.8rem;}.main-title p{margin:4px 0 0;font-size:.95rem;opacity:.85;}
.info-box{background:#e8f4fd;border-left:4px solid #1a73e8;padding:12px 16px;border-radius:6px;margin:10px 0;}
.ok-box{background:#e8f8f0;border-left:4px solid #0d6e3f;padding:12px 16px;border-radius:6px;margin:10px 0;}
.stTabs [data-baseweb="tab"]{font-size:15px;font-weight:500;}
</style>""",unsafe_allow_html=True)

if "data" not in st.session_state:
    st.session_state.data=load_data()

st.markdown("""<div class="main-title"><h1>🌍 Інтелектуальна обробка даних в РІС</h1>
<p>Предметна область: Пріоритетні країни для академічної мобільності студентів</p></div>""",unsafe_allow_html=True)

tabs=st.tabs(["🏠 Про роботу","🗳️ Голосування ЛР1","📊 Результати ЛР1",
              "📋 Евристики (голосування)","⚙️ Застосування евристик",
              "🧬 Генетичний алгоритм","🔐 Адмін"])

# --- TAB 0 ---
with tabs[0]:
    st.header("Про систему")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("📚 Лабораторна робота №1")
        st.markdown("""**Мета:** Дослідити процедуру попереднього преференційного голосування.\n\n**Умови:**\n- Учасник вводить **своє ім'я** (не список!)\n- Обирає **ТОП-3 країни** за пріоритетністю\n- Голосування **анонімне** — ім'я хешується у токен\n- Доступ **розподілений** — через посилання\n- Результат: **ядро лідерів A^λ**""")
        st.subheader("🗂️ 20 країн-об'єктів")
        for i,obj in enumerate(OBJECTS,1): st.write(f"{i:>2}. {FLAG.get(obj,'')} {obj}")
    with c2:
        st.subheader("🔬 Лабораторна робота №2")
        st.markdown("""**Мета:** Евристичне звуження до ≤ 10 об'єктів + генетичний алгоритм.\n\n**Умови:**\n- Відкрите голосування за 2–3 евристики\n- Застосування евристик за популярністю\n- Фінальна підмножина: **≤ 10 країн**\n- **Генетичний алгоритм** (OX-crossover)\n\n**Евристики E1–E7:**""")
        for k,v in HEURISTICS.items(): st.write(f"**{k}:** {v}")

# --- TAB 1 ---
with tabs[1]:
    st.header("🗳️ Голосування — Лабораторна робота №1")
    st.markdown('<div class="info-box">Введіть своє <strong>ім\'я та прізвище</strong>. Воно хешується у анонімний токен — відкрито не зберігається. Оберіть ТОП-3 країни.</div>',unsafe_allow_html=True)
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
            st.subheader("Оберіть ТОП-3 країни:")
            v1=st.selectbox("🥇 1-е місце:",["— Оберіть —"]+OBJECTS,key="v1")
            opts2=[o for o in OBJECTS if o!=v1]
            v2=st.selectbox("🥈 2-е місце:",["— Оберіть —"]+opts2,key="v2")
            opts3=[o for o in opts2 if o!=v2]
            v3=st.selectbox("🥉 3-є місце:",["— Оберіть —"]+opts3,key="v3")
            if st.button("✅ Підтвердити голос",type="primary"):
                if "— Оберіть —" in [v1,v2,v3]: st.error("Оберіть усі 3 позиції!")
                elif len({v1,v2,v3})<3: st.error("Усі три країни мають бути різними!")
                else:
                    DATA["lab1_votes"][token]={"expert":expert,"choices":[v1,v2,v3],"timestamp":datetime.now().isoformat()}
                    save_data(DATA); st.session_state.data=DATA; st.balloons()
                    st.success(f"✅ Голос збережено! Токен: `{token}`"); st.rerun()

# --- TAB 2 ---
with tabs[2]:
    st.header("📊 Результати — Лабораторна робота №1")
    DATA=load_data(); st.session_state.data=DATA; votes=DATA["lab1_votes"]
    m1,m2,m3=st.columns(3)
    m1.metric("Проголосувало",len(votes))
    m2.metric("Країн у голосуванні",len(OBJECTS))
    m3.metric("Країн в ядрі A^λ",len(get_nucleus(votes)) if votes else 0)
    if not votes:
        st.info("Поки немає жодного голосу.")
    else:
        vc=vote_counts(votes); pc=position_counts(votes); nucleus=get_nucleus(votes)
        st.subheader("📋 Анонімні бюлетені")
        rows=[{"Токен":tkn,"🥇 1-е":vd["choices"][0],"🥈 2-е":vd["choices"][1],"🥉 3-є":vd["choices"][2],"Час":vd.get("timestamp","")[:16]} for tkn,vd in votes.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True)
        st.subheader("🏆 Зведена таблиця")
        tbl=[{"Країна":f"{FLAG.get(o,'')} {o}","🥇":pc[o][1],"🥈":pc[o][2],"🥉":pc[o][3],"Всього":vc[o]} for o in OBJECTS if vc.get(o,0)>0]
        df=pd.DataFrame(tbl).sort_values("Всього",ascending=False).reset_index(drop=True)
        st.dataframe(df,use_container_width=True)
        st.subheader("📈 Діаграма голосів")
        objs=[r["Країна"].split(" ",1)[1] if " " in r["Країна"] else r["Країна"] for _,r in df.iterrows()]
        p1v=[pc[o][1] for o in objs]; p2v=[pc[o][2] for o in objs]; p3v=[pc[o][3] for o in objs]
        x=np.arange(len(objs)); fig,ax=plt.subplots(figsize=(13,5))
        ax.bar(x,p1v,.65,label="🥇 1-е",color="#FFD700")
        ax.bar(x,p2v,.65,bottom=p1v,label="🥈 2-е",color="#A8A8A8")
        ax.bar(x,p3v,.65,bottom=[a+b for a,b in zip(p1v,p2v)],label="🥉 3-є",color="#CD7F32")
        ax.set_xticks(x); ax.set_xticklabels(objs,rotation=42,ha="right",fontsize=10)
        ax.set_ylabel("Голосів"); ax.set_title("Розподіл голосів (ЛР1)"); ax.legend()
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); plt.tight_layout(); st.pyplot(fig); plt.close()
        st.subheader(f"🎯 Ядро лідерів A^λ — {len(nucleus)} країн")
        st.markdown('<div class="ok-box">Об\'єднання підмножин претендентів, що увійшли до МП хоча б одного експерта (формула 2).</div>',unsafe_allow_html=True)
        cols=st.columns(5)
        for i,obj in enumerate(sorted(nucleus)): cols[i%5].success(f"{FLAG.get(obj,'')} {obj}")
        if not DATA.get("lab2_final_objects"):
            DATA["lab2_final_objects"]=nucleus; save_data(DATA); st.session_state.data=DATA

# --- TAB 3 ---
with tabs[3]:
    st.header("📋 Голосування за евристики — Лабораторна робота №2")
    st.markdown('<div class="info-box">Відкрите голосування. Введіть ім\'я та оберіть <strong>2–3 евристики</strong>.</div>',unsafe_allow_html=True)
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
        st.markdown("---"); st.subheader("📊 Підрахунок голосів")
        hc={k:0 for k in HEURISTICS}
        for hv in DATA["lab2_heuristic_votes"].values():
            for h in hv:
                if h in hc: hc[h]+=1
        hdf=pd.DataFrame([{"Евристика":k,"Опис":HEURISTICS[k],"Голосів":hc[k]} for k in HEURISTICS]).sort_values("Голосів",ascending=False).reset_index(drop=True)
        st.dataframe(hdf,use_container_width=True)
        fig2,ax2=plt.subplots(figsize=(10,4))
        ax2.barh(hdf["Евристика"][::-1],hdf["Голосів"][::-1],color="#1a73e8")
        ax2.set_xlabel("Голосів"); ax2.set_title("Пріоритетність евристик"); ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True)); plt.tight_layout(); st.pyplot(fig2); plt.close()
        st.subheader("📄 Протокол")
        st.dataframe(pd.DataFrame([{"Учасник":e,"Обрані евристики":", ".join(hv)} for e,hv in DATA["lab2_heuristic_votes"].items()]),use_container_width=True)

# --- TAB 4 ---
with tabs[4]:
    st.header("⚙️ Застосування евристик — Лабораторна робота №2")
    DATA=load_data(); st.session_state.data=DATA; votes=DATA["lab1_votes"]; nucleus=get_nucleus(votes) if votes else []
    if not nucleus:
        st.warning("Спочатку проведіть голосування ЛР1.")
    else:
        st.write(f"**Початкове ядро A^λ:** {len(nucleus)} країн — {', '.join(nucleus)}")
        hc={k:0 for k in HEURISTICS}
        for hv in DATA["lab2_heuristic_votes"].values():
            for h in hv:
                if h in hc: hc[h]+=1
        priority=sorted([k for k in HEURISTICS if hc[k]>0],key=lambda k:hc[k],reverse=True) or list(HEURISTICS.keys())
        st.subheader("🔢 Порядок застосування")
        for i,h in enumerate(priority,1): st.write(f"{i}. **{h}** — {HEURISTICS[h]}  *({hc[h]} голос(ів))*")
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
        st.subheader("📉 Графік зменшення")
        sizes=[s["n"] for s in steps]; labels=[s["Крок"] for s in steps]
        fig3,ax3=plt.subplots(figsize=(9,4))
        ax3.plot(labels,sizes,"o-",color="#1a3a6b",linewidth=2,markersize=8); ax3.axhline(10,color="red",linestyle="--",label="Межа ≤ 10")
        ax3.set_ylabel("Країн"); ax3.set_title("Зменшення підмножини"); ax3.set_xticks(range(len(labels))); ax3.set_xticklabels(labels,rotation=30,ha="right"); ax3.legend(); ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); plt.tight_layout(); st.pyplot(fig3); plt.close()
        st.subheader(f"🏁 Фінальна підмножина: {len(current)} країн")
        for i,obj in enumerate(current,1): st.write(f"{i}. {FLAG.get(obj,'')} {obj}")
        if st.button("💾 Зберегти для Генетичного алгоритму",type="primary"):
            DATA["lab2_final_objects"]=current; save_data(DATA); st.session_state.data=DATA; st.success("✅ Збережено!")

# --- TAB 5 ---
with tabs[5]:
    st.header("🧬 Генетичний алгоритм — Лабораторна робота №2")
    DATA=load_data(); st.session_state.data=DATA; final_objs=DATA.get("lab2_final_objects",[])
    if not final_objs:
        st.warning("Спочатку збережіть фінальну підмножину на вкладці '⚙️ Застосування евристик'.")
    else:
        st.write(f"**Вхідна підмножина:** {len(final_objs)} країн — {', '.join(final_objs)}")
        with st.expander("📖 Опис алгоритму",expanded=False):
            st.markdown("**Задача:** знайти перестановку фінальної підмножини, найбільш узгоджену з МП усіх експертів.\n\n**Функція пристосованості:** за кожну правильно впорядковану пару бал (вага: 1-е місце → +2, 2-е → +1).\n\n| Оператор | Опис |\n|---|---|\n| Ініціалізація | Випадкові перестановки |\n| Схрещування | **Order Crossover (OX)** |\n| Мутація | Обмін двох позицій з ймовірністю p |\n| Селекція | Елітний відбір 50% найкращих |")
        st.subheader("⚙️ Параметри")
        col1,col2,col3=st.columns(3)
        pop_size=col1.slider("Розмір популяції",20,300,80,10)
        generations=col2.slider("Поколінь",20,1000,200,20)
        mut_rate=col3.slider("Мутація p",0.01,0.50,0.10,0.01)
        if st.button("▶️ Запустити",type="primary"):
            with st.spinner("⏳ Виконується..."):
                result,history=genetic_algorithm(final_objs,DATA["lab1_votes"],pop_size,generations,mut_rate)
            DATA["ga_result"]=result; save_data(DATA); st.session_state.data=DATA; st.success("✅ Готово!")
            st.subheader("🏆 Фінальне ранжування:")
            medals=["🥇","🥈","🥉"]
            for i,obj in enumerate(result,1):
                m=medals[i-1] if i<=3 else f"**{i}.**"; st.markdown(f"{m} {FLAG.get(obj,'')} {obj}")
            fig4,ax4=plt.subplots(figsize=(10,4)); ax4.plot(history,color="#0d6e3f",linewidth=1.5)
            ax4.set_xlabel("Покоління"); ax4.set_ylabel("Пристосованість"); ax4.set_title("Збіжність генетичного алгоритму"); plt.tight_layout(); st.pyplot(fig4); plt.close()
            vc=vote_counts(DATA["lab1_votes"]); scores=[vc.get(o,0) for o in result]
            colors=["#FFD700" if i==0 else "#C0C0C0" if i==1 else "#CD7F32" if i==2 else "#4DABF7" for i in range(len(result))]
            y_pos=list(range(len(result)-1,-1,-1)); fig5,ax5=plt.subplots(figsize=(10,5))
            ax5.barh(y_pos,scores,color=colors); ax5.set_yticks(y_pos); ax5.set_yticklabels([f"{FLAG.get(o,'')} {o}" for o in result])
            ax5.set_xlabel("Голосів"); ax5.set_title("Фінальне ранжування"); ax5.xaxis.set_major_locator(plt.MaxNLocator(integer=True)); plt.tight_layout(); st.pyplot(fig5); plt.close()
        elif DATA.get("ga_result"):
            st.subheader("🏆 Результат попереднього запуску:")
            medals=["🥇","🥈","🥉"]
            for i,obj in enumerate(DATA["ga_result"],1):
                m=medals[i-1] if i<=3 else f"**{i}.**"; st.markdown(f"{m} {FLAG.get(obj,'')} {obj}")
            st.info("Натисніть '▶️ Запустити', щоб оновити.")

# --- TAB 6 ADMIN ---
with tabs[6]:
    st.header("🔐 Адміністративна панель")
    st.caption("Тільки для викладача. Пароль за замовчуванням: `admin2024`")
    pwd=st.text_input("🔑 Пароль:",type="password",key="admin_pwd")
    if not pwd:
        st.info("Введіть пароль для доступу.")
    elif hashlib.sha256(pwd.encode()).hexdigest()!=ADMIN_HASH:
        st.error("❌ Невірний пароль")
    else:
        st.success("✅ Доступ надано")
        DATA=load_data(); votes=DATA["lab1_votes"]; heur_votes=DATA["lab2_heuristic_votes"]
        m1,m2,m3,m4=st.columns(4)
        m1.metric("Голосів ЛР1",len(votes))
        m2.metric("Голосів за евристики",len(heur_votes))
        m3.metric("Об'єктів (фінал)",len(DATA.get("lab2_final_objects",[])))
        m4.metric("ГА запускався","Так ✔" if DATA.get("ga_result") else "Ні")
        st.markdown("---")
        st.subheader("📋 Повний протокол ЛР1 (з іменами)")
        if votes:
            prot=[{"Токен":tkn,"Ім'я":vd.get("expert","—"),"🥇 1-е":vd["choices"][0],"🥈 2-е":vd["choices"][1],"🥉 3-є":vd["choices"][2],"Час":vd.get("timestamp","")[:19]} for tkn,vd in votes.items()]
            st.dataframe(pd.DataFrame(prot),use_container_width=True)
        else:
            st.info("Ще немає голосів ЛР1.")
        st.markdown("---")
        st.subheader("📋 Протокол евристик (ЛР2)")
        if heur_votes:
            st.dataframe(pd.DataFrame([{"Учасник":e,"Евристики":", ".join(hv)} for e,hv in heur_votes.items()]),use_container_width=True)
        else:
            st.info("Ще немає голосів за евристики.")
        st.markdown("---")
        st.subheader("🏁 Фінальна підмножина")
        fo=DATA.get("lab2_final_objects",[])
        if fo: st.write(", ".join([f"{FLAG.get(o,'')} {o}" for o in fo]))
        else: st.info("Ще не збережена.")
        st.subheader("🧬 Результат ГА")
        ga=DATA.get("ga_result",[])
        if ga:
            medals=["🥇","🥈","🥉"]
            for i,obj in enumerate(ga,1):
                m=medals[i-1] if i<=3 else f"{i}."; st.write(f"{m} {FLAG.get(obj,'')} {obj}")
        else:
            st.info("ГА ще не запускався.")
        st.markdown("---")
        st.subheader("🗑️ Управління даними")
        c1,c2,c3=st.columns(3)
        if c1.button("🗑️ Очистити голоси ЛР1"):
            DATA["lab1_votes"]={};DATA["lab2_final_objects"]=[];save_data(DATA);st.session_state.data=DATA;st.success("Очищено.");st.rerun()
        if c2.button("🗑️ Очистити евристики"):
            DATA["lab2_heuristic_votes"]={};save_data(DATA);st.session_state.data=DATA;st.success("Очищено.");st.rerun()
        if c3.button("⚠️ Скинути ВСІ дані"):
            fresh={"lab1_votes":{},"lab2_heuristic_votes":{},"lab2_final_objects":[],"ga_result":[]};save_data(fresh);st.session_state.data=fresh;st.warning("Скинуто!");st.rerun()
        st.markdown("---")
        st.subheader("📥 Завантажити дані")
        st.download_button("⬇️ Завантажити data.json",data=json.dumps(DATA,ensure_ascii=False,indent=2).encode("utf-8"),file_name="voting_data.json",mime="application/json")
