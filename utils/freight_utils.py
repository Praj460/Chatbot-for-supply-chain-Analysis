import pandas as pd
import numpy as np
import re

def clean_freight_cost_column_with_id_priority(df):
    df = df.copy()

    id_lookup = {}
    asn_lookup = {}

    if "ID" in df.columns and "Freight Cost (USD)" in df.columns:
        temp_id = df[["ID", "Freight Cost (USD)"]].dropna()
        temp_id = temp_id[temp_id["Freight Cost (USD)"].apply(lambda x: isinstance(x, (int, float, np.number)))]
        id_lookup = dict(zip(temp_id["ID"], temp_id["Freight Cost (USD)"]))

    if "ASN/DN #" in df.columns and "Freight Cost (USD)" in df.columns:
        temp_asn = df[["ASN/DN #", "Freight Cost (USD)"]].dropna()
        temp_asn["ASN/DN #"] = temp_asn["ASN/DN #"].astype(str)
        temp_asn = temp_asn[temp_asn["Freight Cost (USD)"].apply(lambda x: isinstance(x, (int, float, np.number)))]
        asn_lookup = dict(zip(temp_asn["ASN/DN #"], temp_asn["Freight Cost (USD)"]))

    def process_freight(x):
        if pd.isnull(x):
            return np.nan
        elif isinstance(x, (int, float, np.number)):
            return x
        elif isinstance(x, str):
            x_lower = x.lower()
            if "freight included" in x_lower:
                return 0
            elif "invoiced separately" in x_lower:
                return np.nan
            elif "see asn" in x_lower:
                id_match = re.search(r"id#[:\s]*(\d+)", x_lower)
                asn_match = re.search(r"(asn-\d+)", x_lower)

                if id_match:
                    id_number = int(id_match.group(1))
                    if id_number in id_lookup:
                        return id_lookup[id_number]

                if asn_match:
                    asn_number = asn_match.group(1).upper()
                    if asn_number in asn_lookup:
                        return asn_lookup[asn_number]

                return np.nan
            else:
                try:
                    return float(x)
                except:
                    return np.nan
        else:
            return np.nan

    df["Freight Cost (USD)"] = df["Freight Cost (USD)"].apply(process_freight)

    if df["Freight Cost (USD)"].isnull().sum() > 0:
        median_value = df["Freight Cost (USD)"].median()
        df["Freight Cost (USD)"].fillna(median_value, inplace=True)

    return df

