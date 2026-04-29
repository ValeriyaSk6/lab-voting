import streamlit as st
import json
import os
import hashlib
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

OBJECTS = [
    "Україна", "США", "Велика Британія", "Польща", "Франція",
    "Німеччина", "Канада", "Японія", "Австралія", "Нідерланди",
    "Швеція", "Норвегія", "Данія", "Фінляндія", "Швейцарія",
    "Чехія", "Австрія", "Іспанія", "Португалія", "Естонія"
]

EXPERTS = [
    "Андрієнко Марія", "Бойко Олексій", "Василенко Тетяна", "Гнатюк Роман",
    "Данченко Оксана", "Євтушенко Ігор", "Захарченко Наталія", "Іванов Сергій",
    "Кравченко Людмила", "Лисенко Дмитро", "Мельник Катерина", "Науменко Олег",
    "Олексієнко Вікторія", "Петренко Андрій", "Романенко Юлія", "Сидоренко Максим",
    "Тимченко Ірина", "Усенко Богдан", "Федченко Олена", "Харченко Василь",
    "Викладач"
]

HEURISTICS = {
    "Е1": "Участь лише в одному МП на 3-му місці",
    "Е2": "Участь лише в одному МП на 2-му місці",
    "Е3": "Участь лише в одному МП на 1-му місці",
    "Е4": "Участь лише у 2-х МП на 3-му місці",
    "Е5": "Участь в одному МП на 3-му та одному МП на 2-му місці",
    "Е6": "Загальна кількість згадувань < 2 (авторська евристика)",
    "Е7": "Жодного разу не отримала 1-е місце (авторська евристика)",
}

ADMIN_PASSWORD_HASH = hashlib.sha256("admin2024".encode()).hexdigest()
DATA_FILE = "data.json"


# ============================================================
# DATA MANAGEMENT
# ============================================================

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "lab1_votes": {},
        "lab2_heuristic_votes": {},
        "lab2_final_objects": [],
        "ga_result": [],
    }


def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_token(expert_name: str) -> str:
    """Generate an 8-char anonymous token for Lab 1."""
    return hashlib.sha256(f"iod_ris_lab1_{expert_name}_2024".encode()).hexdigest()[:8]


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def vote_counts(lab1_votes: dict) -> dict:
    counts = {o: 0 for o in OBJECTS}
    for vd in lab1_votes.values():
        for obj in vd.get("choices", []):
            if obj in counts:
                counts[obj] += 1
    return counts


def position_counts(lab1_votes: dict) -> dict:
    pc = {o: {1: 0, 2: 0, 3: 0} for o in OBJECTS}
    for vd in lab1_votes.values():
        for pos, obj in enumerate(vd.get("choices", []), 1):
            if obj in pc and pos in (1, 2, 3):
                pc[obj][pos] += 1
    return pc


def get_nucleus(lab1_votes: dict) -> list:
    nucleus = set()
    for vd in lab1_votes.values():
        for obj in vd.get("choices", []):
            nucleus.add(obj)
    return sorted(nucleus)


def apply_heuristic(objects: list, lab1_votes: dict, key: str):
    """Return (remaining, removed) after applying one heuristic."""
    vc = vote_counts(lab1_votes)
    pc = position_counts(lab1_votes)
    to_remove = []
    for obj in objects:
        total = vc.get(obj, 0)
        p1 = pc.get(obj, {1: 0})[1]
        p2 = pc.get(obj, {2: 0})[2]
        p3 = pc.get(obj, {3: 0})[3]

        if key == "Е1" and total == 1 and p3 == 1 and p1 == 0 and p2 == 0:
            to_remove.append(obj)
        elif key == "Е2" and total == 1 and p2 == 1 and p1 == 0 and p3 == 0:
            to_remove.append(obj)
        elif key == "Е3" and total == 1 and p1 == 1 and p2 == 0 and p3 == 0:
            to_remove.append(obj)
        elif key == "Е4" and total == 2 and p3 == 2 and p1 == 0 and p2 == 0:
            to_remove.append(obj)
        elif key == "Е5" and p3 == 1 and p2 == 1 and p1 == 0:
            to_remove.append(obj)
        elif key == "Е6" and total < 2:
            to_remove.append(obj)
        elif key == "Е7" and p1 == 0:
            to_remove.append(obj)

    remaining = [o for o in objects if o not in to_remove]
    return remaining, to_remove


# ============================================================
# GENETIC ALGORITHM
# ============================================================

def fitness(permutation: list, lab1_votes: dict) -> float:
    """Higher = better agreement with expert MPs."""
    score = 0.0
    for vd in lab1_votes.values():
        choices = [c for c in vd.get("choices", []) if c in permutation]
        for i in range(len(choices)):
            for j in range(i + 1, len(choices)):
                # choices[i] should rank BEFORE choices[j]
                ri = permutation.index(choices[i])
                rj = permutation.index(choices[j])
                if ri < rj:
                    score += 3 - i  # weight: pos 1 = 2, pos 2 = 1
    return score


def ox_crossover(p1: list, p2: list) -> list:
    """Order Crossover (OX)."""
    n = len(p1)
    if n < 2:
        return p1[:]
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a : b + 1] = p1[a : b + 1]
    remaining = [x for x in p2 if x not in child]
    j = 0
    for i in range(n):
        if child[i] is None:
            child[i] = remaining[j]
            j += 1
    return child


def genetic_algorithm(
    objects: list,
    lab1_votes: dict,
    pop_size: int = 50,
    generations: int = 100,
    mut_rate: float = 0.1,
) -> tuple:
    n = len(objects)
    if n == 0:
        return [], []

    population = [random.sample(objects, n) for _ in range(pop_size)]
    best_ind = population[0][:]
    best_fit = fitness(best_ind, lab1_votes)
    history = []

    for gen in range(generations):
        scored = sorted(
            [(ind, fitness(ind, lab1_votes)) for ind in population],
            key=lambda x: x[1],
            reverse=True,
        )
        if scored[0][1] > best_fit:
            best_fit = scored[0][1]
            best_ind = scored[0][0][:]

        history.append(scored[0][1])
        survivors = [ind for ind, _ in scored[: pop_size // 2]]

        new_pop = survivors[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = ox_crossover(p1, p2)
            new_pop.append(child)

        # Mutation
        for i in range(1, len(new_pop)):
            if random.random() < mut_rate:
                i1, i2 = random.sample(range(n), 2)
                new_pop[i][i1], new_pop[i][i2] = new_pop[i][i2], new_pop[i][i1]

        population = new_pop

    return best_ind, history


# ============================================================
# STREAMLIT PAGE CONFIG & STYLES
# ============================================================

st.set_page_config(
    page_title="ІОД РІС — Лабораторні роботи №1 і №2",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    /* Main header */
    .main-title {
        background: linear-gradient(135deg, #1a3a6b 0%, #0d6e3f 100%);
        color: white; padding: 22px 28px; border-radius: 12px;
        margin-bottom: 24px; text-align: center;
    }
    .main-title h1 { margin: 0; font-size: 1.8rem; }
    .main-title p  { margin: 4px 0 0; font-size: 0.95rem; opacity: 0.85; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f0f4ff; border-radius: 8px; padding: 10px 16px;
    }

    /* Nice info boxes */
    .info-box {
        background: #e8f4fd; border-left: 4px solid #1a73e8;
        padding: 12px 16px; border-radius: 6px; margin: 12px 0;
    }
    .success-box {
        background: #e8f8f0; border-left: 4px solid #0d6e3f;
        padding: 12px 16px; border-radius: 6px; margin: 12px 0;
    }

    /* Tab font */
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; }

    /* Subtle divider */
    hr { border: none; border-top: 1px solid #e0e0e0; margin: 20px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# LOAD DATA
# ============================================================

if "data" not in st.session_state:
    st.session_state.data = load_data()

DATA = st.session_state.data

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
  <h1>🌍 Інтелектуальна обробка даних в РІС</h1>
  <p>Предметна область: Пріоритетні країни для академічної мобільності студентів</p>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# TABS
# ============================================================

tab_labels = [
    "🏠 Про роботу",
    "🗳️ Голосування ЛР1",
    "📊 Результати ЛР1",
    "📋 Евристики (голосування)",
    "⚙️ Застосування евристик",
    "🧬 Генетичний алгоритм",
    "🔐 Адмін",
]

tabs = st.tabs(tab_labels)


# ============================================================
# TAB 0 — ABOUT
# ============================================================

with tabs[0]:
    st.header("Про систему")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📚 Лабораторна робота №1 — Технології розподіленого введення даних")
        st.markdown("""
**Мета:** Запропонувати та дослідити процедуру попереднього преференційного голосування для множини з кількох десятків об'єктів.

**Умови:**
- Кожен експерт обирає **ТОП-3 країни** у порядку пріоритетності (множинне порівняння)
- Голосування **анонімне** — кожен отримує випадковий токен
- Доступ **розподілений** — через веб-браузер
- Результат: **ядро лідерів A^λ** — об'єднання підмножин МП всіх експертів
""")

        st.subheader("📋 Список об'єктів (n = 20 країн)")
        for i, obj in enumerate(OBJECTS, 1):
            flag_map = {
                "Україна": "🇺🇦", "США": "🇺🇸", "Велика Британія": "🇬🇧",
                "Польща": "🇵🇱", "Франція": "🇫🇷", "Німеччина": "🇩🇪",
                "Канада": "🇨🇦", "Японія": "🇯🇵", "Австралія": "🇦🇺",
                "Нідерланди": "🇳🇱", "Швеція": "🇸🇪", "Норвегія": "🇳🇴",
                "Данія": "🇩🇰", "Фінляндія": "🇫🇮", "Швейцарія": "🇨🇭",
                "Чехія": "🇨🇿", "Австрія": "🇦🇹", "Іспанія": "🇪🇸",
                "Португалія": "🇵🇹", "Естонія": "🇪🇪",
            }
            st.write(f"{i:>2}. {flag_map.get(obj, '')} {obj}")

    with col2:
        st.subheader("🔬 Лабораторна робота №2 — Технології розподіленої обробки даних")
        st.markdown("""
**Мета:** Евристичне звуження підмножини до ≤ 10 об'єктів, реалізація еволюційного алгоритму ранжування.

**Умови:**
- Відкрите голосування: 2–3 евристики без порівняння
- Застосування евристик у порядку пріоритетності (за голосами)
- Фінальна підмножина: **≤ 10 країн**
- Генетичний алгоритм для ранжування переможців

**Евристики E1–E7:**
""")
        for k, v in HEURISTICS.items():
            st.write(f"**{k}:** {v}")

        st.markdown("---")
        st.subheader("👥 Список експертів (k = 21)")
        for i, exp in enumerate(EXPERTS, 1):
            icon = "👨‍🏫" if exp == "Викладач" else "👤"
            st.write(f"{i:>2}. {icon} {exp}")


# ============================================================
# TAB 1 — LAB 1 VOTING
# ============================================================

with tabs[1]:
    st.header("🗳️ Голосування — Лабораторна робота №1")

    st.markdown(
        '<div class="info-box">Кожен експерт обирає <strong>3 країни</strong> у порядку пріоритету '
        "(1-е місце = найвищий пріоритет). Голосування анонімне.</div>",
        unsafe_allow_html=True,
    )

    expert = st.selectbox("Оберіть свій ідентифікатор:", EXPERTS, key="voting_expert")
    token = get_token(expert)
    st.caption(f"Ваш анонімний токен (зберігається замість імені): `{token}`")

    votes = DATA["lab1_votes"]

    if token in votes:
        st.success("✅ Ви вже проголосували. Ваш голос збережено!")
        ch = votes[token]["choices"]
        st.write(f"🥇 1-е місце: **{ch[0]}**")
        st.write(f"🥈 2-е місце: **{ch[1]}**")
        st.write(f"🥉 3-є місце: **{ch[2]}**")
        if st.button("🔄 Переголосувати"):
            del DATA["lab1_votes"][token]
            save_data(DATA)
            st.rerun()
    else:
        st.subheader("Вкажіть ТОП-3 країни у порядку пріоритетності:")

        c1 = st.selectbox("🥇 1-е місце (найвищий пріоритет):", ["— Оберіть —"] + OBJECTS, key="c1")
        opts2 = [o for o in OBJECTS if o != c1]
        c2 = st.selectbox("🥈 2-е місце:", ["— Оберіть —"] + opts2, key="c2")
        opts3 = [o for o in opts2 if o != c2]
        c3 = st.selectbox("🥉 3-є місце:", ["— Оберіть —"] + opts3, key="c3")

        if st.button("✅ Підтвердити голос", type="primary"):
            if "— Оберіть —" in [c1, c2, c3]:
                st.error("Будь ласка, оберіть усі 3 позиції!")
            elif len({c1, c2, c3}) < 3:
                st.error("Усі 3 позиції мають бути різними країнами!")
            else:
                DATA["lab1_votes"][token] = {
                    "expert": expert,
                    "choices": [c1, c2, c3],
                    "timestamp": datetime.now().isoformat(),
                }
                save_data(DATA)
                st.balloons()
                st.success(f"✅ Голос збережено! Ваш токен: `{token}`")
                st.rerun()


# ============================================================
# TAB 2 — LAB 1 RESULTS
# ============================================================

with tabs[2]:
    st.header("📊 Результати — Лабораторна робота №1")

    # Reload fresh data
    DATA = load_data()
    st.session_state.data = DATA

    votes = DATA["lab1_votes"]
    n_votes = len(votes)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Проголосувало", f"{n_votes}")
    col_m2.metric("Очікується", f"{len(EXPERTS)}")
    col_m3.metric("Залишилось", f"{len(EXPERTS) - n_votes}")

    if n_votes == 0:
        st.info("Поки немає голосів. Зачекайте, поки експерти проголосують.")
        st.stop()

    vc = vote_counts(votes)
    pc = position_counts(votes)
    nucleus = get_nucleus(votes)

    # ---- Anonymous ballots ----
    st.subheader("📋 Бюлетені (анонімні)")
    ballot_rows = []
    for tkn, vd in votes.items():
        ch = vd["choices"]
        ballot_rows.append({"Токен": tkn, "🥇 1-е місце": ch[0], "🥈 2-е місце": ch[1], "🥉 3-є місце": ch[2]})
    st.dataframe(pd.DataFrame(ballot_rows), use_container_width=True)

    # ---- Vote count table ----
    st.subheader("🏆 Зведена таблиця голосів")
    count_rows = []
    for obj in OBJECTS:
        total = vc.get(obj, 0)
        if total > 0:
            count_rows.append({
                "Країна": obj,
                "🥇 1-е місце": pc[obj][1],
                "🥈 2-е місце": pc[obj][2],
                "🥉 3-є місце": pc[obj][3],
                "Всього": total,
            })
    count_df = pd.DataFrame(count_rows).sort_values("Всього", ascending=False).reset_index(drop=True)
    st.dataframe(count_df, use_container_width=True)

    # ---- Bar chart ----
    st.subheader("📈 Діаграма популярності країн")
    sorted_objs = count_df["Країна"].tolist()
    p1_vals = [pc[o][1] for o in sorted_objs]
    p2_vals = [pc[o][2] for o in sorted_objs]
    p3_vals = [pc[o][3] for o in sorted_objs]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(sorted_objs))
    w = 0.6
    ax.bar(x, p1_vals, w, label="1-е місце 🥇", color="#FFD700")
    ax.bar(x, p2_vals, w, bottom=p1_vals, label="2-е місце 🥈", color="#A8A8A8")
    ax.bar(x, p3_vals, w, bottom=[a + b for a, b in zip(p1_vals, p2_vals)], label="3-є місце 🥉", color="#CD7F32")
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_objs, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Кількість голосів")
    ax.set_title("Розподіл голосів за країнами та місцями (ЛР1)", fontsize=13)
    ax.legend()
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ---- Nucleus ----
    st.subheader(f"🎯 Ядро лідерів A^λ — {len(nucleus)} країн")
    st.markdown(
        '<div class="success-box">Об\'єднання підмножин претендентів, що увійшли до МП '
        "хоча б одного експерта (формула 2 із завдання).</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    for i, obj in enumerate(sorted(nucleus)):
        cols[i % 5].success(obj)

    # Save nucleus if not yet saved
    if not DATA.get("lab2_final_objects"):
        DATA["lab2_final_objects"] = nucleus
        save_data(DATA)
        st.session_state.data = DATA


# ============================================================
# TAB 3 — LAB 2 HEURISTICS VOTING
# ============================================================

with tabs[3]:
    st.header("📋 Голосування за евристики — Лабораторна робота №2")

    DATA = load_data()
    st.session_state.data = DATA

    st.markdown(
        '<div class="info-box">Відкрите голосування. Оберіть <strong>2–3 евристики</strong>, '
        "які слід застосувати першими для відсіювання найменш підтриманих об'єктів.</div>",
        unsafe_allow_html=True,
    )

    exp2 = st.selectbox("Ваш ідентифікатор:", EXPERTS, key="exp2")

    if exp2 in DATA["lab2_heuristic_votes"]:
        st.success(f"✅ Ви вже обрали евристики: **{', '.join(DATA['lab2_heuristic_votes'][exp2])}**")
        if st.button("🔄 Змінити вибір"):
            del DATA["lab2_heuristic_votes"][exp2]
            save_data(DATA)
            st.session_state.data = DATA
            st.rerun()
    else:
        st.subheader("Список евристик — оберіть 2–3:")
        selected_h = []
        for key, desc in HEURISTICS.items():
            if st.checkbox(f"**{key}** — {desc}", key=f"heur_{key}"):
                selected_h.append(key)

        st.write(f"Обрано: **{len(selected_h)}** евристики(к)")

        if st.button("✅ Зберегти вибір евристик", type="primary"):
            if len(selected_h) < 2 or len(selected_h) > 3:
                st.error("Будь ласка, оберіть від 2 до 3 евристик!")
            else:
                DATA["lab2_heuristic_votes"][exp2] = selected_h
                save_data(DATA)
                st.session_state.data = DATA
                st.success("✅ Вибір збережено!")
                st.rerun()

    # ---- Heuristics results ----
    if DATA["lab2_heuristic_votes"]:
        st.markdown("---")
        st.subheader("📊 Підрахунок голосів за евристиками")

        h_counts = {k: 0 for k in HEURISTICS}
        for hvotes in DATA["lab2_heuristic_votes"].values():
            for h in hvotes:
                if h in h_counts:
                    h_counts[h] += 1

        h_df = (
            pd.DataFrame([
                {"Евристика": k, "Опис": HEURISTICS[k], "Голосів": h_counts[k]}
                for k in HEURISTICS
            ])
            .sort_values("Голосів", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(h_df, use_container_width=True)

        # Bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.barh(h_df["Евристика"][::-1], h_df["Голосів"][::-1], color="#1a73e8")
        ax2.set_xlabel("Кількість голосів")
        ax2.set_title("Пріоритетність евристик (за голосами експертів)")
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.subheader("📄 Протокол голосування за евристиками")
        prot_rows = [{"Експерт": exp, "Обрані евристики": ", ".join(hvs)}
                     for exp, hvs in DATA["lab2_heuristic_votes"].items()]
        st.dataframe(pd.DataFrame(prot_rows), use_container_width=True)


# ============================================================
# TAB 4 — APPLY HEURISTICS
# ============================================================

with tabs[4]:
    st.header("⚙️ Застосування евристик — Лабораторна робота №2")

    DATA = load_data()
    st.session_state.data = DATA

    votes = DATA["lab1_votes"]
    nucleus = get_nucleus(votes) if votes else []

    if not nucleus:
        st.warning("Спочатку проведіть голосування ЛР1 та перегляньте результати (вкладка 'Результати ЛР1').")
        st.stop()

    st.write(f"**Початкове ядро лідерів A^λ:** {len(nucleus)} країн")
    st.write(", ".join(nucleus))

    # Determine heuristic priority
    if DATA["lab2_heuristic_votes"]:
        h_counts = {k: 0 for k in HEURISTICS}
        for hvotes in DATA["lab2_heuristic_votes"].values():
            for h in hvotes:
                if h in h_counts:
                    h_counts[h] += 1
        priority = sorted([k for k in HEURISTICS if h_counts[k] > 0], key=lambda k: h_counts[k], reverse=True)
    else:
        priority = list(HEURISTICS.keys())
        st.info("Голосування за евристики ще не проводилось — застосовуємо в стандартному порядку E1→E7.")

    st.subheader("🔢 Порядок застосування евристик (за пріоритетом)")
    for i, h in enumerate(priority, 1):
        cnt = DATA["lab2_heuristic_votes"] and sum(
            1 for v in DATA["lab2_heuristic_votes"].values() if h in v
        )
        st.write(f"{i}. **{h}** — {HEURISTICS[h]} ({cnt} голос(ів))")

    st.markdown("---")
    st.subheader("🔄 Покрокове застосування евристик")

    current = nucleus[:]
    steps = [{"Крок": "Початкове ядро", "Залишилось": len(current),
               "Країни": ", ".join(current), "Видалено": "—"}]

    for h in priority:
        if len(current) <= 10:
            break
        new_set, removed = apply_heuristic(current, votes, h)
        if removed:
            current = new_set
            steps.append({
                "Крок": f"Після {h}",
                "Залишилось": len(current),
                "Країни": ", ".join(current),
                "Видалено": ", ".join(removed),
            })

    for step in steps:
        color = "🟢" if step["Залишилось"] <= 10 else "🟡"
        with st.expander(f"{color} {step['Крок']}: залишилось **{step['Залишилось']}** країн", expanded=True):
            if step["Видалено"] != "—":
                st.error(f"❌ Видалено: {step['Видалено']}")
            st.success(f"✅ Залишилось: {step['Країни']}")

    # Size reduction chart
    st.subheader("📉 Графік зменшення підмножини")
    sizes = [s["Залишилось"] for s in steps]
    labels = [s["Крок"] for s in steps]
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(labels, sizes, "o-", color="#1a3a6b", linewidth=2, markersize=8)
    ax3.axhline(10, color="red", linestyle="--", label="Межа ≤ 10")
    ax3.set_ylabel("Кількість країн")
    ax3.set_title("Зменшення підмножини об'єктів при застосуванні евристик")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=30, ha="right")
    ax3.legend()
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    st.subheader(f"🏁 Фінальна підмножина: {len(current)} країн")
    for i, obj in enumerate(current, 1):
        st.write(f"{i}. {obj}")

    if st.button("💾 Зберегти фінальну підмножину для ГА", type="primary"):
        DATA["lab2_final_objects"] = current
        save_data(DATA)
        st.session_state.data = DATA
        st.success("✅ Збережено! Тепер запустіть Генетичний алгоритм.")


# ============================================================
# TAB 5 — GENETIC ALGORITHM
# ============================================================

with tabs[5]:
    st.header("🧬 Генетичний алгоритм — Лабораторна робота №2")

    DATA = load_data()
    st.session_state.data = DATA

    final_objs = DATA.get("lab2_final_objects", [])

    if not final_objs:
        st.warning("Спочатку збережіть фінальну підмножину на вкладці 'Застосування евристик'.")
        st.stop()

    st.write(f"**Вхідна підмножина для ранжування:** {len(final_objs)} країн")
    st.write(", ".join(final_objs))

    # Algorithm description
    with st.expander("📖 Опис генетичного алгоритму", expanded=False):
        st.markdown("""
**Постановка задачі:**  
Знайти перестановку (ранжування) об'єктів фінальної підмножини, найбільш узгоджену з множинними порівняннями (МП) усіх експертів.

**Функція пристосованості:**  
Для кожної пари (a_i ≻ a_j) у МП експерта нараховується бал, якщо у поточній перестановці a_i стоїть перед a_j.  
Вага балу залежить від місця: 1-е місце (вага 2), 2-е місце (вага 1).

**Оператори:**
- **Ініціалізація:** Випадкові перестановки
- **Схрещування:** Order Crossover (OX) — зберігає відносний порядок
- **Мутація:** Обмін двох елементів з ймовірністю `p_mut`
- **Селекція:** Елітний відбір 50% найкращих особин
""")

    st.subheader("⚙️ Параметри алгоритму")
    col1, col2, col3 = st.columns(3)
    pop_size = col1.slider("Розмір популяції", 20, 300, 80, 10)
    generations = col2.slider("Кількість поколінь", 20, 1000, 200, 20)
    mut_rate = col3.slider("Ймовірність мутації", 0.01, 0.50, 0.10, 0.01)

    if st.button("▶️ Запустити генетичний алгоритм", type="primary"):
        with st.spinner("⏳ Виконується генетичний алгоритм..."):
            result, history = genetic_algorithm(
                final_objs, DATA["lab1_votes"], pop_size, generations, mut_rate
            )
        DATA["ga_result"] = result
        save_data(DATA)
        st.session_state.data = DATA
        st.success("✅ Алгоритм завершено!")

        # Show ranking
        st.subheader("🏆 Результат ранжування:")
        medals = ["🥇", "🥈", "🥉"]
        for i, obj in enumerate(result, 1):
            m = medals[i - 1] if i <= 3 else f"**{i}.**"
            st.markdown(f"{m} {obj}")

        # Fitness history
        st.subheader("📈 Графік збіжності алгоритму")
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(history, color="#0d6e3f", linewidth=1.5)
        ax4.set_xlabel("Покоління")
        ax4.set_ylabel("Найкраща пристосованість")
        ax4.set_title("Збіжність генетичного алгоритму")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

        # Bar chart
        vc = vote_counts(DATA["lab1_votes"])
        st.subheader("📊 Фінальне ранжування з підтримкою")
        fig5, ax5 = plt.subplots(figsize=(10, 5))
        scores = [vc.get(o, 0) for o in result]
        colors = ["#FFD700" if i == 0 else "#C0C0C0" if i == 1 else "#CD7F32" if i == 2 else "#4DABF7"
                  for i in range(len(result))]
        y_pos = range(len(result) - 1, -1, -1)
        ax5.barh(list(y_pos), scores, color=colors)
        ax5.set_yticks(list(y_pos))
        ax5.set_yticklabels(result)
        ax5.set_xlabel("Загальна кількість голосів (підтримка)")
        ax5.set_title("Фінальне ранжування країн (Генетичний алгоритм)")
        ax5.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close()

    elif DATA.get("ga_result"):
        st.subheader("🏆 Попередній результат (з останнього запуску):")
        medals = ["🥇", "🥈", "🥉"]
        for i, obj in enumerate(DATA["ga_result"], 1):
            m = medals[i - 1] if i <= 3 else f"**{i}.**"
            st.markdown(f"{m} {obj}")
        st.info("Натисніть '▶️ Запустити' вище, щоб отримати новий результат.")


# ============================================================
# TAB 6 — ADMIN
# ============================================================

with tabs[6]:
    st.header("🔐 Адміністративна панель")
    st.caption("Доступно тільки для викладача. Пароль за замовчуванням: `admin2024`")

    pwd = st.text_input("Пароль:", type="password", key="admin_pwd")
    if not pwd:
        st.stop()

    if hashlib.sha256(pwd.encode()).hexdigest() != ADMIN_PASSWORD_HASH:
        st.error("❌ Невірний пароль")
        st.stop()

    st.success("✅ Доступ надано")

    DATA = load_data()
    votes = DATA["lab1_votes"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Голосів ЛР1", len(votes))
    col2.metric("Голосів за евристики", len(DATA["lab2_heuristic_votes"]))
    col3.metric("Об'єктів (фінал)", len(DATA.get("lab2_final_objects", [])))
    col4.metric("ГА запускався", "Так" if DATA.get("ga_result") else "Ні")

    # Full protocol with names
    st.subheader("📋 Повний протокол ЛР1 (конфіденційно)")
    if votes:
        prot = []
        for tkn, vd in votes.items():
            ch = vd["choices"]
            prot.append({
                "Токен": tkn,
                "Ім'я": vd.get("expert", "—"),
                "🥇 1-е місце": ch[0],
                "🥈 2-е місце": ch[1],
                "🥉 3-є місце": ch[2],
                "Час": vd.get("timestamp", "")[:19],
            })
        st.dataframe(pd.DataFrame(prot), use_container_width=True)
    else:
        st.info("Немає голосів ЛР1.")

    # Хто ще не проголосував
    st.subheader("⏳ Хто ще не проголосував (ЛР1):")
    voted_tokens = set(votes.keys())
    not_voted = [exp for exp in EXPERTS if get_token(exp) not in voted_tokens]
    if not_voted:
        for exp in not_voted:
            st.write(f"• {exp}")
    else:
        st.success("Усі проголосували!")

    st.markdown("---")
    st.subheader("🗑️ Управління даними")
    c1, c2, c3 = st.columns(3)
    if c1.button("🗑️ Очистити голоси ЛР1"):
        DATA["lab1_votes"] = {}
        save_data(DATA)
        st.session_state.data = DATA
        st.success("Голоси ЛР1 очищено.")
        st.rerun()
    if c2.button("🗑️ Очистити евристики"):
        DATA["lab2_heuristic_votes"] = {}
        save_data(DATA)
        st.session_state.data = DATA
        st.success("Голоси за евристики очищено.")
        st.rerun()
    if c3.button("🗑️ Скинути всі дані", type="secondary"):
        fresh = {"lab1_votes": {}, "lab2_heuristic_votes": {}, "lab2_final_objects": [], "ga_result": []}
        save_data(fresh)
        st.session_state.data = fresh
        st.warning("Всі дані скинуто!")
        st.rerun()

    # Download data as JSON
    st.subheader("📥 Завантажити дані")
    json_str = json.dumps(DATA, ensure_ascii=False, indent=2)
    st.download_button(
        "⬇️ Завантажити data.json",
        data=json_str.encode("utf-8"),
        file_name="voting_data.json",
        mime="application/json",
    )
