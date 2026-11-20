"""Compare original Plan 2 with a concentrated aggressive momentum variant.
Uses same historical NIFTY 500 membership, adjusted Yahoo prices, month-end
rebalance dates and 0.50% turnover cost. Research only; no orders are placed.
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
MEMBERSHIP_FILE = "nifty500_membership_timeline.csv"
CAPITAL, COST = 20000.0, 0.005
TOP100, TOP50, AGG_N = 100, 50, 5
TRAIN_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")

m = pd.read_csv(MEMBERSHIP_FILE)
m["effective_date"] = pd.to_datetime(m["effective_date"], errors="coerce")
m["symbol"] = m["symbol"].astype(str).str.strip().str.upper()
m = m.dropna(subset=["effective_date"])
m = m[m.symbol.str.match(r"^[A-Z0-9]+$")].drop_duplicates(["effective_date","symbol"])
m = m.sort_values(["effective_date","symbol"])
def ys(s): return str(s)+".NS"
def members(date):
    x=m[m.effective_date<=date]
    if x.empty: return []
    d=x.effective_date.max()
    return sorted({ys(s) for s in x.loc[x.effective_date==d,"symbol"]})

symbols=sorted({ys(s) for s in m.symbol})
start=m.effective_date.min()-pd.DateOffset(years=2)
end=pd.Timestamp.today().normalize()+pd.Timedelta(days=1)
frames=[]
for i in range(0,len(symbols),25):
    batch=symbols[i:i+25]
    for attempt in range(3):
        try:
            d=yf.download(batch,start=start.strftime("%Y-%m-%d"),end=end.strftime("%Y-%m-%d"),
                          auto_adjust=True,progress=False,threads=False,group_by="column")
            if d is None or d.empty: continue
            if isinstance(d.columns,pd.MultiIndex):
                if "Close" not in d.columns.get_level_values(0): continue
                c=d["Close"]
            else:
                if "Close" not in d.columns: continue
                c=d[["Close"]].copy(); c.columns=[batch[0]]
            frames.append(c); break
        except Exception: pass
if not frames: raise RuntimeError("No Yahoo price data. Check workflow network/logs.")
prices=pd.concat(frames,axis=1,sort=True)
prices=prices.loc[:,~prices.columns.duplicated()].sort_index().dropna(axis=1,how="all")
if prices.empty: raise RuntimeError("Downloaded price matrix is empty.")
last_day=prices.index.max().normalize()
last_complete=(last_day.to_period("M")-1).to_timestamp("M")
monthly=prices.resample("ME").last()
monthly=monthly.loc[monthly.index<=last_complete].dropna(how="all")
dates=[d for d in monthly.index if d>=m.effective_date.min()+pd.DateOffset(months=13)]
if len(dates)<3: raise RuntimeError("Not enough completed monthly prices.")

def before(s,date):
    x=s.loc[s.index<=date].dropna()
    return None if x.empty else float(x.iloc[-1])

def original_signal(sym,date):
    s=prices[sym].dropna()
    p0=before(s,date-pd.DateOffset(months=12))
    p1=before(s,date-pd.DateOffset(months=2))
    if p0 is None or p1 is None or p0<=0: return None
    path=s.loc[(s.index>=date-pd.DateOffset(months=12))&(s.index<=date-pd.DateOffset(months=2))]
    if len(path)<100: return None
    dr=path.pct_change().dropna()
    dr=dr[dr!=0]
    if dr.empty: return None
    ret=p1/p0-1
    return ret,(1 if ret>0 else -1)*((dr<0).mean()-(dr>0).mean())

def aggressive_signal(sym,date):
    s=prices[sym].dropna()
    hist=s.loc[:date]
    if len(hist)<252: return None
    p=before(s,date)
    p1=before(s,date-pd.DateOffset(months=1))
    p3=before(s,date-pd.DateOffset(months=3))
    p6=before(s,date-pd.DateOffset(months=6))
    p12=before(s,date-pd.DateOffset(months=12))
    if any(x is None or x<=0 for x in [p,p1,p3,p6,p12]): return None
    sma50,sma200=hist.tail(50).mean(),hist.tail(200).mean()
    if not (p>sma50>sma200): return None
    r3,r6,r12=p/p3-1,p/p6-1,p/p12-1
    r1=p/p1-1
    vol=hist.tail(253).pct_change().dropna().std()*np.sqrt(252)
    score=(0.30*r3+0.30*r6+0.40*r12-0.25*r1)/vol if vol>0 else np.nan
    return score if np.isfinite(score) and score>0 else None

def run(kind):
    prev={}; rows=[]; audit=[]; skipped=0
    for i,date in enumerate(dates[:-1]):
        nxt=dates[i+1]
        cand=[s for s in members(date) if s in prices.columns]
        sig={}
        if kind=="original":
            for s in cand:
                z=original_signal(s,date)
                if z is not None: sig[s]=z
            if len(sig)<TOP100: skipped+=1; continue
            first=sorted(sig.items(),key=lambda x:x[1][0],reverse=True)[:TOP100]
            selected=[s for s,z in sorted(first,key=lambda x:x[1][1])[:TOP50]]
        else:
            for s in cand:
                z=aggressive_signal(s,date)
                if z is not None: sig[s]=z
            if len(sig)<AGG_N: skipped+=1; continue
            selected=[s for s,z in sorted(sig.items(),key=lambda x:x[1],reverse=True)[:AGG_N]]
        w=1/len(selected); new={s:w for s in selected}
        turnover=sum(abs(new.get(s,0)-prev.get(s,0)) for s in set(new)|set(prev))
        gross=0.0
        for s,weight in new.items():
            p0=monthly.at[date,s] if s in monthly.columns else np.nan
            p1=monthly.at[nxt,s] if s in monthly.columns else np.nan
            if pd.notna(p0) and pd.notna(p1) and p0>0: gross+=weight*(p1/p0-1)
            audit.append({"strategy":kind,"rebalance_date":date,"symbol":s})
        rows.append({"return_date":nxt,"monthly_return":(1-turnover*COST)*(1+gross)-1})
        prev=new
    return pd.DataFrame(rows),pd.DataFrame(audit),skipped

def stats(r):
    r=pd.Series(r).dropna()
    if len(r)<2:return None
    eq=(1+r).cumprod(); vol=r.std(ddof=1)*np.sqrt(12)
    return {"months":len(r),"total_return":eq.iloc[-1]-1,
            "cagr":eq.iloc[-1]**(12/len(r))-1,"annual_volatility":vol,
            "sharpe_rf0":r.mean()/r.std(ddof=1)*np.sqrt(12) if r.std(ddof=1)>0 else np.nan,
            "max_drawdown":(eq/eq.cummax()-1).min(),"winning_months":(r>0).mean()}

base,ba,bs=run("original")
agg,aa,ass=run("aggressive")
if base.empty or agg.empty: raise RuntimeError("No results for one strategy; see skipped months.")
cmp=base.merge(agg,on="return_date",suffixes=("_original","_aggressive"))
cmp=cmp.sort_values("return_date")
if cmp.empty: raise RuntimeError("Strategies have no overlapping return dates.")
summary=[]
for period,frame in [("TRAIN",cmp[cmp.return_date<=TRAIN_END]),
                     ("UNSEEN_TEST",cmp[cmp.return_date>=TEST_START]),("FULL_OVERLAP",cmp)]:
    for col,label in [("monthly_return_original","Original Plan 2"),
                      ("monthly_return_aggressive","Aggressive 5-stock")]:
        z=stats(frame[col])
        if z:
            summary.append({"period":period,"strategy":label,**z,
                            "ending_value_on_20000":CAPITAL*(1+z["total_return"])})
            print(period,label,{k:(round(v,4) if isinstance(v,(float,np.floating)) else v) for k,v in z.items()},
                  "₹20k ending value:",round(CAPITAL*(1+z["total_return"]),2))
pd.DataFrame(summary).to_csv("aggressive_vs_plan2_summary.csv",index=False)
cmp.to_csv("aggressive_vs_plan2_monthly.csv",index=False)
ba.to_csv("aggressive_original_selection_audit.csv",index=False)
aa.to_csv("aggressive_new_selection_audit.csv",index=False)
print("Skipped months:",{"original":bs,"aggressive":ass})
print("Price symbols:",prices.shape[1],"last complete month:",last_complete.date())
