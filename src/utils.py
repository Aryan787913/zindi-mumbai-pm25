import pandas as pd, re
def parse_ist(s):
    """Parse mixed IST timestamp strings -> tz-aware Asia/Kolkata."""
    s=pd.Series(s).astype(str)
    out=pd.Series(pd.NaT,index=s.index,dtype='datetime64[ns, Asia/Kolkata]')
    iso=s.str.match(r'^\d{4}-\d{2}-\d{2}')
    if iso.any(): out[iso]=pd.to_datetime(s[iso],format='ISO8601',utc=True).dt.tz_convert('Asia/Kolkata')
    h=~iso
    if h.any():
        t=s[h].str.replace(r'\s*IST.*$','',regex=True).str.replace(r'^[A-Za-z]{3}\s+','',regex=True)
        out[h]=pd.to_datetime(t,format='%Y-%m-%d %I:%M %p').dt.tz_localize('Asia/Kolkata')
    return out
