"""
UFI Project — Tableau Integration Module

Since direct Tableau Server connections require authentication,
this module provides THREE ways to connect your UFI data to Tableau:

METHOD 1  ── CSV Export (always works)
    Exports clean, analysis-ready CSVs that Tableau Desktop can
    connect to natively via Data → Connect → Text File.

METHOD 2  ── Tableau Hyper Extract (.hyper) — needs tableauhyperapi
    Creates a native Tableau extract for fast in-memory analytics.
    Install:  pip install tableauhyperapi
    Only available on Windows/macOS/Linux x86_64.

METHOD 3  ── Tableau Workbook Template (.twb XML)
    Generates a pre-configured .twb XML file pointing at the CSV.
    Open in Tableau Desktop → all sheets and data source pre-wired.

This file implements Method 1 (always) and Method 3 (always),
and attempts Method 2 if the library is installed.

TABLEAU DASHBOARD SPECS (build these in Tableau Desktop)
─────────────────────────────────────────────────────────
Sheet 1 — City UFI Over Time
    Cols: hour  |  Rows: AVG(ufi_score)  |  Mark: Line

Sheet 2 — Neighbourhood Heatmap
    Cols: neighbourhood  |  Rows: hour  |  Color: AVG(ufi_score)
    Mark: Square  |  Color palette: Red-Green Diverging

Sheet 3 — Component Breakdown
    Rows: neighbourhood  |  Cols: measure names
    Values: C1, C2, C3, C4  |  Mark: Bar (grouped)

Sheet 4 — UFI Class Distribution
    Cols: ufi_class  |  Rows: COUNT(road_id)  |  Mark: Bar

Sheet 5 — Road Map (if geometry available)
    Use road_name as detail, ufi_score as colour.

Dashboard
    Arrange all 5 sheets with a filter action on neighbourhood.
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("tableau")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Method 1: CSV Exports ──────────────────────────────────────────────────────

def export_csvs(df: pd.DataFrame):
    """Export four analysis-ready CSVs for Tableau."""

    # 1. Full scored dataset
    full_path = OUTPUT_DIR / "ufi_full.csv"
    df.to_csv(full_path, index=False)
    print(f"Exported: {full_path}  ({len(df)} rows)")

    # 2. Neighbourhood summary (by hour)
    nbhd_hourly = (
        df.groupby(["neighbourhood", "hour"])
          .agg(
              avg_ufi=("ufi_score", "mean"),
              avg_C1 =("C1", "mean"),
              avg_C2 =("C2", "mean"),
              avg_C3 =("C3", "mean"),
              avg_C4 =("C4", "mean"),
              incident_rate=("incident_count", "mean"),
          )
          .reset_index()
    )
    path = OUTPUT_DIR / "ufi_neighbourhood_hourly.csv"
    nbhd_hourly.to_csv(path, index=False)
    print(f"Exported: {path}  ({len(nbhd_hourly)} rows)")

    # 3. Peak hour snapshot (temporal_weight == 1.0)
    peak = df[df["temporal_weight"] == 1.0].copy()
    path = OUTPUT_DIR / "ufi_peak_hours.csv"
    peak.to_csv(path, index=False)
    print(f"Exported: {path}  ({len(peak)} rows)")

    # 4. Road-level aggregated stats
    road_agg = (
        df.groupby(["road_id", "road_name", "neighbourhood"])
          .agg(
              avg_ufi       =("ufi_score", "mean"),
              peak_ufi      =("ufi_score", "max"),
              dominant_class=("ufi_class", lambda x: x.mode()[0]),
              avg_speed     =("avg_speed", "mean"),
              avg_volume    =("volume", "mean"),
              avg_C3        =("C3", "mean"),
          )
          .reset_index()
    )
    path = OUTPUT_DIR / "ufi_road_summary.csv"
    road_agg.to_csv(path, index=False)
    print(f"Exported: {path}  ({len(road_agg)} rows)")

    return full_path


# ── Method 2: Tableau Hyper Extract (optional) ────────────────────────────────

def export_hyper(df: pd.DataFrame):
    """
    Create a Tableau .hyper extract.
    Requires:  pip install tableauhyperapi
    """
    try:
        from tableauhyperapi import (
            HyperProcess, Telemetry, Connection, CreateMode,
            TableDefinition, SqlType, Inserter, TableName
        )
    except ImportError:
        print("\n[Hyper] tableauhyperapi not installed. Skipping .hyper export.")
        print("  To enable: pip install tableauhyperapi")
        return

    hyper_path = OUTPUT_DIR / "ufi_data.hyper"

    type_map = {
        "road_id":         SqlType.text(),
        "road_name":       SqlType.text(),
        "neighbourhood":   SqlType.text(),
        "hour":            SqlType.int(),
        "avg_speed":       SqlType.double(),
        "volume":          SqlType.int(),
        "capacity":        SqlType.int(),
        "ufi_score":       SqlType.double(),
        "ufi_class":       SqlType.text(),
        "C1":              SqlType.double(),
        "C2":              SqlType.double(),
        "C3":              SqlType.double(),
        "C4":              SqlType.double(),
        "temporal_weight": SqlType.double(),
        "nlp_severity":    SqlType.double(),
        "incident_count":  SqlType.int(),
    }

    cols_to_export = list(type_map.keys())
    table_def = TableDefinition(
        table_name=TableName("Extract", "UFI"),
        columns=[TableDefinition.Column(c, t) for c, t in type_map.items()]
    )

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hp:
        with Connection(hp.endpoint, str(hyper_path), CreateMode.CREATE_AND_REPLACE) as conn:
            conn.catalog.create_schema_if_not_exists("Extract")
            conn.catalog.create_table(table_def)
            with Inserter(conn, table_def) as inserter:
                for row in df[cols_to_export].itertuples(index=False):
                    inserter.add_row(list(row))
                inserter.execute()

    print(f"\n[Hyper] Exported: {hyper_path}  ({len(df)} rows)")


# ── Method 3: Tableau Workbook XML (.twb) ─────────────────────────────────────

def export_twb(csv_path: Path):
    """
    Generate a .twb XML that Tableau Desktop can open directly.
    The workbook pre-configures the data source to point at ufi_full.csv.
    """
    twb_path = OUTPUT_DIR / "UFI_Dashboard.twb"
    csv_abs   = str(csv_path.resolve()).replace("\\", "/")

    twb_xml = f"""<?xml version='1.0' encoding='utf-8' ?>
<workbook source-build='2024.1.0' source-platform='linux' version='21.4' xmlns:user='http://www.tableausoftware.com/xml/user'>
  <datasources>
    <datasource caption='UFI Data' inline='true' name='ufi_full' version='21.4'>
      <connection class='textscan' filename='{csv_abs}' separator=',' locale='en_US'>
        <relation name='ufi_full' table='[ufi_full#csv]' type='text'>
          <columns header='yes' separator=','>
            <column datatype='string'  name='road_id'/>
            <column datatype='string'  name='road_name'/>
            <column datatype='string'  name='neighbourhood'/>
            <column datatype='integer' name='hour'/>
            <column datatype='real'    name='avg_speed'/>
            <column datatype='integer' name='volume'/>
            <column datatype='integer' name='capacity'/>
            <column datatype='real'    name='ufi_score'/>
            <column datatype='string'  name='ufi_class'/>
            <column datatype='real'    name='C1'/>
            <column datatype='real'    name='C2'/>
            <column datatype='real'    name='C3'/>
            <column datatype='real'    name='C4'/>
            <column datatype='real'    name='temporal_weight'/>
            <column datatype='real'    name='nlp_severity'/>
            <column datatype='integer' name='incident_count'/>
            <column datatype='string'  name='incident_text'/>
          </columns>
        </relation>
      </connection>
      <!-- Calculated fields -->
      <column caption='UFI Score Bucket' datatype='string' name='[UFI Score Bucket]' role='dimension' type='nominal'>
        <calculation class='tableau' formula='IF [ufi_score] &lt; 15 THEN "Free Flow" ELSEIF [ufi_score] &lt; 35 THEN "Low" ELSEIF [ufi_score] &lt; 55 THEN "Moderate" ELSEIF [ufi_score] &lt; 75 THEN "High" ELSE "Severe" END'/>
      </column>
      <column caption='Is Peak Hour' datatype='boolean' name='[Is Peak Hour]' role='dimension' type='nominal'>
        <calculation class='tableau' formula='([hour] >= 7 AND [hour] &lt;= 10) OR ([hour] >= 17 AND [hour] &lt;= 20)'/>
      </column>
      <column caption='Congestion Level (numeric)' datatype='real' name='[Congestion Level]' role='measure' type='quantitative'>
        <calculation class='tableau' formula='IF [ufi_score] &lt; 15 THEN 1 ELSEIF [ufi_score] &lt; 35 THEN 2 ELSEIF [ufi_score] &lt; 55 THEN 3 ELSEIF [ufi_score] &lt; 75 THEN 4 ELSE 5 END'/>
      </column>
    </datasource>
  </datasources>
  <worksheets>
    <!-- Sheet 1: Hourly UFI trend -->
    <worksheet name='City UFI by Hour'>
      <table>
        <view>
          <datasources>
            <datasource caption='UFI Data' name='ufi_full'/>
          </datasources>
          <aggregation value='true'/>
        </view>
        <style/>
        <panes>
          <pane>
            <mark class='Line'/>
            <encodings>
              <column>[hour]</column>
              <row>AVG([ufi_score])</row>
            </encodings>
          </pane>
        </panes>
      </table>
    </worksheet>
    <!-- Sheet 2: Neighbourhood heatmap -->
    <worksheet name='Neighbourhood Heatmap'>
      <table>
        <view>
          <datasources>
            <datasource caption='UFI Data' name='ufi_full'/>
          </datasources>
          <aggregation value='true'/>
        </view>
        <panes>
          <pane>
            <mark class='Square'/>
            <encodings>
              <column>[neighbourhood]</column>
              <row>[hour]</row>
              <color>AVG([ufi_score])</color>
            </encodings>
          </pane>
        </panes>
      </table>
    </worksheet>
  </worksheets>
  <windows>
    <window classname='worksheet' maximized='true' name='City UFI by Hour'/>
  </windows>
</workbook>
"""
    twb_path.write_text(twb_xml, encoding="utf-8")
    print(f"\n[TWB] Exported Tableau workbook: {twb_path}")
    print("  → Open in Tableau Desktop to see pre-wired sheets.")
    print("  → Update the CSV path if you move the file.")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_tableau_export(df: pd.DataFrame):
    print("\n══════════════════════════════════════")
    print("   TABLEAU EXPORT")
    print("══════════════════════════════════════")

    csv_path = export_csvs(df)
    export_hyper(df)
    export_twb(csv_path)

    print(f"\nAll Tableau files in: {OUTPUT_DIR}")
    print("\n── How to connect in Tableau Desktop ──")
    print("  1. File → Open → UFI_Dashboard.twb   (Method 3 – instant)")
    print("  2. Data → Connect → Text File → ufi_full.csv  (Method 1 – manual)")
    print("  3. Data → Connect → Tableau Extract → ufi_data.hyper  (Method 2 – fastest)")


if __name__ == "__main__":
    df = pd.read_csv("data/ufi_scored.csv")
    run_tableau_export(df)
