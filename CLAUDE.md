# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a data science academic project for UW MSIS 522 (Data Science Workflow). The repository contains VAERS (Vaccine Adverse Event Reporting System) data — COVID-19 adverse event reports from December 2020 through June 2024.

## Data Schema

Three CSV files linked by `VAERS_ID`:

**VAERSDATA.csv** (~877 MB, ~1M rows) — main adverse event reports:
- Demographics: `AGE_YRS`, `SEX`, `STATE`
- Outcomes: `DIED`, `L_THREAT`, `ER_VISIT`, `HOSPITAL`, `HOSPDAYS`, `DISABLE`, `RECOVD`
- Dates: `RECVDATE`, `VAX_DATE`, `ONSET_DATE`, `DATEDIED`; `NUMDAYS` = onset-to-report lag
- Free text: `SYMPTOM_TEXT`, `HISTORY`, `OTHER_MEDS`, `LAB_DATA`, `CUR_ILL`

**VAERSSYMPTOMS.csv** (~100 MB, ~1.36M rows) — coded symptoms (MedDRA terms):
- `VAERS_ID` + up to 5 symptom/version pairs (`SYMPTOM1`–`SYMPTOM5`, `SYMPTOMVERSION1`–`SYMPTOMVERSION5`)
- One-to-many with VAERSDATA (a single report can span multiple rows)

**VAERSVAX.csv** (~76 MB, ~1.07M rows) — vaccine details:
- `VAX_TYPE`, `VAX_MANU`, `VAX_LOT`, `VAX_DOSE_SERIES`, `VAX_ROUTE`, `VAX_SITE`, `VAX_NAME`
- One-to-many with VAERSDATA

## Working with This Data

**Joining tables:** Use `VAERS_ID` as the foreign key. A single `VAERS_ID` in VAERSSYMPTOMS can produce multiple rows (up to 5 symptoms per report, potentially across rows). Aggregate or pivot before joining to VAERSDATA to avoid row explosion.

**Missing values:** Blanks and `"UNK"` both indicate unknown/missing. Boolean outcome fields (`DIED`, `HOSPITAL`, etc.) use `"Y"` for yes; blank means no/unknown.

**Scale considerations:** Files are large enough to require chunked reads or dask/polars instead of loading full DataFrames into memory with pandas.

**Assignment document:** `MSIS_522_HW1_Data_Science_Workflow.docx` contains the specific task requirements.
