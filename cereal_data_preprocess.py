import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d


sdate = "2013-12-1"
edate = "2022-3-1"

data = pd.read_excel('C:/Users/flori/Downloads/cereal_prices.xlsx')

list_countries = []
for country in set(data["Member State"]):
    if country not in list_countries:
        list_countries.append([country, []])
        for product in set(data[data["Member State"] == country]["Product Name"]):
            if product not in list_countries[-1][1]:
                list_countries[-1][1].append(product)



dic = {'date': pd.date_range(start=sdate, end=edate, freq="MS")
       }
df = pd.DataFrame(dic)

lenlist = []
final_countries = []
final_cereal = []

for country in list_countries:
    for cereal in country[1]:

        per = data[(data["Member State"] == country[0]) & (data["Product Name"] == cereal)].drop(
            ["Marketing Year", "Stage Name"], axis=1)["Reference period"].dt.to_period("M")
        g = data[(data["Member State"] == country[0]) & (data["Product Name"] == cereal)].drop(
            ["Marketing Year", "Stage Name"], axis=1)["Price (€/Tonne)"].groupby(per)
        g = g.mean()
        g.index = g.index.to_timestamp()

        if len(g) >= 100:
            final_countries.append(country[0])
            final_cereal.append(cereal)

            # plt.plot(g.index, [price for price in g],'o')
            # plt.show()

            d = {'date': g.index, f'{country[0]} {cereal}': [price for price in g]}
            df2 = pd.DataFrame(d)
            mask = (df2['date'] >= sdate) & (df2['date'] <= edate)
            df2 = df2.loc[mask]
            df = pd.merge(df, df2, on="date", how="outer")


            lenlist.append(len(df2["date"]))


final_countries = list(set(final_countries))
final_cereal = list(set(final_cereal))

for i in range(len(final_cereal)):
    if ' ' in final_cereal[i]:
        final_cereal[i] = final_cereal[i].rpartition(' ')[-1]

final_cereal = list(set(final_cereal))

for i in df.drop(['date'], axis=1):
    if df[i].isnull().values.any():
        print("price",i, df[i].isnull().values.sum())
        y = df[df[i].isnull() == False][i]
        x = y.index
        f = interp1d(x, y, kind="linear")
        xnew = np.linspace(0, 99, num=100, endpoint=True)

        df[i] = f(xnew)
        # plt.figure()
        # plt.plot(x,y,'o')
        # plt.plot(xnew,f(xnew),'x')
        # plt.show()

for country in final_countries:
    c = []
    for j in df:
        if j.startswith(country):
            c.append(j)
    for cereal in final_cereal:
        matching = [s for s in c if s.endswith(cereal)]
        if len(matching) >= 2:
            df[f'{country} {cereal}'] = df[matching].mean(axis=1)
            df = df.drop(columns=matching)

trade_data = pd.read_csv('C:/Users/flori/Downloads/EU_CEREALS_trade_data_en.csv')
trade_data["price"] = 1000 * trade_data['Value in thousand euro'] / trade_data['Quantity in tonnes (grain equivalent)']
trade_data = trade_data.drop(
    columns=['Value in thousand euro', 'Quantity in tonnes (grain equivalent)', 'Product Code (CN)', 'Partner',
             'Month Order in MY', 'Month'])

trade_data = trade_data.drop(trade_data[trade_data['Product Group'] == "Other cereals"].index)
trade_data = trade_data.drop(trade_data[trade_data['Product Group'] == "Sorghum"].index)
trade_data = trade_data.drop(trade_data[trade_data['Product Group'] == "Triticale"].index)


for country in list(set(trade_data["Member State"])):
    if country not in final_countries:
        trade_data = trade_data.drop(trade_data[trade_data["Member State"] == country].index)


trade_data['Month Date'] = pd.to_datetime(trade_data['Month Date'], dayfirst=True)

export_data = trade_data[trade_data["Flow"] == "EXPORT"]
import_data = trade_data[trade_data["Flow"] == "IMPORT"]

export_countries = []
for country in set(export_data["Member State"]):
    if country not in export_countries:
        export_countries.append([country, []])
        for product in set(export_data[export_data["Member State"] == country]["Product Group"]):
            if product not in export_countries[-1][1]:
                export_countries[-1][1].append(product)

dic = {'date': pd.date_range(start=sdate, end=edate, freq="MS")
       }
dfexport = pd.DataFrame(dic)

for country in export_countries:
    for cereal in country[1]:
        per = export_data[(export_data["Member State"] == country[0]) & (export_data["Product Group"] == cereal)].drop(
            ["Marketing Year"], axis=1)["Month Date"].dt.to_period("M")
        g = export_data[(export_data["Member State"] == country[0]) & (export_data["Product Group"] == cereal)].drop(
            ["Marketing Year"], axis=1)["price"].groupby(per)
        g = g.mean()
        g.index = g.index.to_timestamp()

        d3 = {'date': g.index, f'{country[0].lower()} {cereal.lower()}': [price for price in g]}
        df3 = pd.DataFrame(d3)
        mask = (df3['date'] >= sdate) & (df3['date'] <= edate)
        df3 = df3.loc[mask]
        dfexport = pd.merge(dfexport, df3, on="date", how="outer")


droplist = [i for i in dfexport.columns[1:] if (
        math.isnan(dfexport[i].iloc[0]) or math.isnan(dfexport[i].iloc[-1]) or math.isinf(
    dfexport[i].iloc[0]) or math.isinf(dfexport[i].iloc[-1]))]
dfexport = dfexport.drop(droplist, axis=1)

droplist = []
droplist = [i for i in dfexport.columns[1:-1] if dfexport[i].isnull().sum() >= 75]
dfexport = dfexport.drop(droplist, axis=1)

for i in dfexport.drop(['date'], axis=1):
    if dfexport[i].isnull().values.any() or dfexport[i].isin([np.inf]).values.any():
        print("export", i, dfexport[i].isnull().values.sum())
        y = dfexport[(dfexport[i].isnull() == False) & (dfexport[i].isin([np.inf]) == False)][i]
        x = y.index
        f = interp1d(x, y, kind="linear")
        xnew = np.linspace(0, 99, num=100, endpoint=True)

        dfexport[i] = f(xnew)

df.columns = df.columns.str.lower()
dfexport.columns = dfexport.columns.str.lower()
for i in range(len(final_cereal)):
    final_cereal[i] = final_cereal[i].lower()

for i in range(len(final_countries)):
    final_countries[i] = final_countries[i].lower()

for country in final_countries:
    c = []
    for j in dfexport:
        if j.startswith(country):
            c.append(j)
    for cereal in final_cereal:
        matching = [s for s in c if s.endswith(cereal)]
        if len(matching) >= 2:
            dfexport[f'{country} {cereal}'] = dfexport[matching].mean(axis=1)
            dfexport = dfexport.drop(columns=matching)

import_countries = []
for country in set(import_data["Member State"]):
    if country not in import_countries:
        import_countries.append([country, []])
        for product in set(import_data[import_data["Member State"] == country]["Product Group"]):
            if product not in import_countries[-1][1]:
                import_countries[-1][1].append(product)

dic = {'date': pd.date_range(start=sdate, end=edate, freq="MS")
       }
dfimport = pd.DataFrame(dic)

for country in import_countries:
    for cereal in country[1]:
        per = import_data[(import_data["Member State"] == country[0]) & (import_data["Product Group"] == cereal)].drop(
            ["Marketing Year"], axis=1)["Month Date"].dt.to_period("M")
        g = import_data[(import_data["Member State"] == country[0]) & (import_data["Product Group"] == cereal)].drop(
            ["Marketing Year"], axis=1)["price"].groupby(per)
        g = g.mean()
        g.index = g.index.to_timestamp()

        d4 = {'date': g.index, f'{country[0].lower()} {cereal.lower()}': [price for price in g]}
        df4 = pd.DataFrame(d4)
        mask = (df4['date'] >= sdate) & (df4['date'] <= edate)
        df4 = df4.loc[mask]
        dfimport = pd.merge(dfimport, df4, on="date", how="outer")


droplist = [i for i in dfimport.columns[1:] if (
        math.isnan(dfimport[i].iloc[0]) or math.isnan(dfimport[i].iloc[-1]) or math.isinf(
    dfimport[i].iloc[0]) or math.isinf(dfimport[i].iloc[-1]))]
dfimport = dfimport.drop(droplist, axis=1)

droplist = []
droplist = [i for i in dfimport.columns[1:-1] if dfimport[i].isnull().sum() >= 75]
dfimport = dfimport.drop(droplist, axis=1)

for i in dfimport.drop(['date'], axis=1):
    if dfimport[i].isnull().values.any() or dfimport[i].isin([np.inf]).values.any():
        print("import", i, dfimport[i].isnull().values.sum())
        y = dfimport[(dfimport[i].isnull() == False) & (dfimport[i].isin([np.inf]) == False)][i]
        x = y.index
        f = interp1d(x, y, kind="linear")
        xnew = np.linspace(0, 99, num=100, endpoint=True)

        dfimport[i] = f(xnew)

dfimport.columns = dfimport.columns.str.lower()

for country in final_countries:
    c = []
    for j in dfimport:
        if j.startswith(country):
            c.append(j)
    for cereal in final_cereal:
        matching = [s for s in c if s.endswith(cereal)]
        if len(matching) >= 2:
            dfimport[f'{country} {cereal}'] = dfimport[matching].mean(axis=1)
            dfimport = dfimport.drop(columns=matching)

for i in range(1, len(dfimport.columns)):
    split = list(dfimport.columns[i].split(' '))
    dfimport = dfimport.rename(columns={dfimport.columns[i]: f"import {split[0]} {split[-1]}"})

for i in range(1, len(df.columns)):
    split = list(df.columns[i].split(' '))
    df = df.rename(columns={df.columns[i]: f"prices {split[0]} {split[-1]}"})

for i in range(1, len(dfexport.columns)):
    split = list(dfexport.columns[i].split(' '))
    dfexport = dfexport.rename(columns={dfexport.columns[i]: f"export {split[0]} {split[-1]}"})

final_df = pd.merge(df, dfimport, on="date", how="outer")
final_df = pd.merge(final_df, dfexport, on="date", how="outer")
final_df=final_df.drop(columns=['date'])

plt.plot(final_df.iloc[:,3])

final_df= final_df.diff(axis=0)
plt.plot(final_df.iloc[:,3])
plt.title(final_df.columns[3])
plt.legend(["Non stationnary data","Data after diff function"])
plt.show()
final_df=final_df.drop(labels=0,axis=0)

#final_df.to_csv("cereal_database_linear.csv", index=False)
#final_df.to_excel("cereal_database_linear.xlsx", index=False)

#df.drop(columns="date").to_csv("cereal_prices_linear.csv", index=False)



'''Prices database: https://agridata.ec.europa.eu/extensions/DashboardCereals/ExtCerealsPrice.html#
    To download the data, select "Data Explorer" in the "Select View" menu, right click on the database and click on "Export data"
    
Import/Export database : https://agridata.ec.europa.eu/extensions/DashboardCereals/CerealsTrade.html#
    To download the data, follow the exact same steps as above.
'''


## excel code used to rearange the processed data if necessary (bugs for instance)
## https://trumpexcel.com/concatenate-excel-ranges/