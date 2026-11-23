import pandas as pd, numpy as np
raw = pd.read_csv('data/online_retail_challenge.csv', dtype=str, keep_default_na=False)
log = {}
df = raw.copy()
for c in df.columns:
    df[c] = df[c].str.strip()
log['raw_rows'] = len(df)

# 1) Text: TransactionType -> two clean labels
type_map = {'purchase': 'Purchase', 'cancellation': 'Cancellation', 'cancelled': 'Cancellation'}
df['TransactionType'] = df['TransactionType'].str.lower().map(type_map)
assert df['TransactionType'].notna().all()

# 2) Quantity: remove unit text, to number
df['Quantity'] = pd.to_numeric(df['Quantity'].str.replace(' pcs', '', regex=False), errors='coerce')
# 3) UnitPrice: remove £, comma decimal -> dot, to number
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'].str.replace('£', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')
assert df['Quantity'].notna().all() and df['UnitPrice'].notna().all()

# 4) InvoiceDate: several formats -> datetime (dayfirst, confirmed by values like 23/12/2010)
def parse_date(x):
    for fmt in ('%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M', '%b %d %Y %H:%M'):   # each format parsed explicitly
        try:
            return pd.to_datetime(x, format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.NaT
df['InvoiceDate'] = df['InvoiceDate'].map(parse_date)
log['bad_dates'] = int(df['InvoiceDate'].isna().sum())

# 5) Country: standardize spelling variants, blanks / Unspecified -> Unknown
country_map = {'eire': 'Ireland', 'ireland': 'Ireland', 'uk': 'United Kingdom', 'united kingdom': 'United Kingdom',
               'germany': 'Germany', 'france': 'France', 'usa': 'USA', 'channel islands': 'Channel Islands',
               'unspecified': 'Unknown', '': 'Unknown'}
df['Country'] = df['Country'].str.lower().map(lambda x: country_map.get(x, None) if x in country_map else None)
# fallback: keep original title-case for any remaining names
raw_country = raw['Country'].str.strip()
df['Country'] = df['Country'].fillna(raw_country.str.strip())
df['Country'] = df['Country'].replace({'': 'Unknown', 'Unspecified': 'Unknown'})

# 6) Remove exact duplicate rows (same invoice, product, qty, price, date...) 
before = len(df)
df = df.drop_duplicates()
log['duplicates_removed'] = before - len(df)

# 7) Invalid prices (negative) -> drop
bad_price = df['UnitPrice'] < 0
log['negative_price_removed'] = int(bad_price.sum())
df = df[~bad_price]

# 8) Consistency check: cancellation flag vs sign of quantity
c_inv = df['InvoiceNo'].str.startswith('C')
assert ((df['TransactionType'] == 'Cancellation') == c_inv).all()
assert (df.loc[df['TransactionType']=='Cancellation','Quantity'] < 0).all()
assert (df.loc[df['TransactionType']=='Purchase','Quantity'] >= 0).all()

# 9) Derived column (after columns are clean)
df['TotalAmount'] = (df['Quantity'] * df['UnitPrice']).round(2)

df = df.reset_index(drop=True)
log['clean_rows'] = len(df)
for k,v in log.items(): print(k, v)
print(df.dtypes.to_string())
print(df['Country'].value_counts().to_string())
print('TransactionType:', df['TransactionType'].value_counts().to_dict())
print('Date range:', df['InvoiceDate'].min(), df['InvoiceDate'].max())
print('Missing:', df.isna().sum().to_dict())
df.to_csv('data/retail_clean.csv', index=False)
