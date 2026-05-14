import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("🚦 Urban Flow Index Dashboard")

# Load data
df = pd.read_csv("data/ufi_scored.csv")

# Sidebar filters
st.sidebar.header("Filters")
neighbourhood = st.sidebar.selectbox("Select Area", df["neighbourhood"].unique())
hour = st.sidebar.slider("Select Hour", 0, 23, (7, 10))

filtered = df[
    (df["neighbourhood"] == neighbourhood) &
    (df["hour"] >= hour[0]) &
    (df["hour"] <= hour[1])
]

# Show table
st.subheader("📋 Data Table")
st.dataframe(filtered.head(50))

# Guard: warn if filter returns no rows
if filtered.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

# Graph 1: Hourly UFI  ← was using df, now uses filtered
st.subheader("📈 UFI by Hour")
hourly = filtered.groupby("hour")["ufi_score"].mean()
fig, ax = plt.subplots()
ax.plot(hourly.index, hourly.values, marker="o")
ax.set_xlabel("Hour")
ax.set_ylabel("UFI Score")
ax.set_title(f"UFI by Hour — {neighbourhood}")
st.pyplot(fig)
plt.close(fig)

# Graph 2: Neighbourhood comparison  ← was using df, now uses hour-filtered slice only
st.subheader("📊 Area Comparison")
# Keep hour filter but show ALL neighbourhoods so comparison is meaningful
hour_filtered = df[(df["hour"] >= hour[0]) & (df["hour"] <= hour[1])]
nb = hour_filtered.groupby("neighbourhood")["ufi_score"].mean().sort_values()
colors = ["crimson" if n == neighbourhood else "steelblue" for n in nb.index]
fig2, ax2 = plt.subplots()
nb.plot(kind="barh", ax=ax2, color=colors)
ax2.set_xlabel("Mean UFI Score")
ax2.set_title(f"Area Comparison (hours {hour[0]}–{hour[1]})")
st.pyplot(fig2)
plt.close(fig2)

# Graph 3: Distribution  ← was using df, now uses filtered
st.subheader("📉 UFI Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(filtered["ufi_score"], bins=30, ax=ax3)
ax3.set_title(f"UFI Distribution — {neighbourhood}")
st.pyplot(fig3)
plt.close(fig3)