import pandas as pd
import numpy as np
import seaborn as sns

def rolling_mean(series):
    return series.fillna(0).rolling(window=3).mean().mean()

def rolling_max(series):
    return series.fillna(0).rolling(window=3).mean().max()

def rolling_min(series):
    return series.fillna(0).rolling(window=3).mean().min()

def rolling_std(series):
    return series.fillna(0).rolling(window=3).mean().std()

def diffmean(s):
    return s.dropna().drop_duplicates().diff().mean()


def generate_features(month_to_predict, max_train_month, features_folder_name):
    items = pd.read_csv("cleaned_data/items.csv")
    items_categories = pd.read_csv("cleaned_data/item_categories.csv")
    sales_train = pd.read_csv("cleaned_data/sales_train.csv")
    shops = pd.read_csv("cleaned_data/shops.csv")

    df_init = pd.merge(items, items_categories, on="item_category_id", how="inner")
    df_init = pd.merge(df_init, sales_train, on="item_id", how="inner")
    df_init = pd.merge(df_init, shops, on="shop_id", how="inner")

    df_aux = df_init.copy()
    df = df_init[df_init["date_block_num"] < month_to_predict]
    assert(df["date_block_num"].max() == max_train_month)
    print("Maximum month in features after processing will be:", df.date_block_num.value_counts().index.max())

    df.drop(columns=["item_name", "item_category_name", "date", "shop_name"], inplace=True)


    df_ts = df.groupby(["item_id", "shop_id", "date_block_num"]).agg({"item_cnt_day":"sum"}).reset_index().rename(columns={"item_cnt_day":"item_cnt_month"})

    all_months = [x for x in range(0, month_to_predict + 1)]

    combinations = []
    for month in all_months:
        all_items = list(sales_train[sales_train.date_block_num == month].item_id.unique())
        all_shops = list(sales_train[sales_train.date_block_num == month].shop_id.unique())

        if month == 34:
            all_items = pd.read_csv("cleaned_data/test.csv")["item_id"].unique()
            all_shops = pd.read_csv("cleaned_data/test.csv")["shop_id"].unique()
        
        for item_id in all_items:
            for shop_id in all_shops:
                combinations.append([item_id, shop_id, month])

    combinations = pd.DataFrame(combinations, columns=["item_id", "shop_id", "date_block_num"])

    df_ts = pd.merge(combinations, df_ts, how="left", on=["item_id", "shop_id", "date_block_num"])
    df_ts["item_cnt_month"].fillna(0, inplace=True)
    df_ts = pd.merge(df_ts, items, on="item_id", how="inner").drop(columns=["item_name"])
    df_ts = pd.merge(df_ts, items_categories, on="item_category_id", how="inner")
    df_ts = pd.merge(df_ts, shops, on="shop_id", how="inner").drop(columns=["shop_name"])
    df_ts.drop("item_category_name", axis=1, inplace=True)
    df_ts["item_cnt_month"].clip(0, 20, inplace=True)

    df_aspects = df_ts[df_ts.date_block_num == month_to_predict]

    df_ts = df_ts[df_ts.date_block_num <= max_train_month]

    #---------------------------------------------------------------------------------------
    '''print("Features of categories")

    ## Stats last n months.

    categories_features = df_aspects[["item_category_id"]].drop_duplicates(subset=["item_category_id"])

    months = [max_train_month - i for i in range(3)]

    for month in months:
        items_sold_in_month = df_ts[df_ts["date_block_num"] == month].groupby("item_category_id").agg({"item_cnt_month":"mean"}).reset_index()
        
        items_sold_in_month.rename(columns={
            "item_cnt_month":"item_category_mean_shifted_" + str(month_to_predict - month) + "_months",
        }, inplace=True)

        categories_features = pd.merge(categories_features, items_sold_in_month, on="item_category_id", how="left")

    categories_features.to_csv("generated/" + features_folder_name + "features_categories.csv", index=False)'''

    #---------------------------------------------------------------------------------------
    '''print("Features of categories types")

    category_types_features = df_aspects[["item_category_type"]].drop_duplicates(subset=["item_category_type"])

    ##Stats last n months.

    months = [max_train_month - i for i in range(3)]

    for month in months:
        items_sold_in_month = df_ts[df_ts["date_block_num"] == month].groupby("item_category_type").agg({"item_cnt_month":"mean"}).reset_index()
        
        items_sold_in_month.rename(columns={
            "item_cnt_month":"item_category_type_mean_shifted_" + str(month_to_predict - month) + "_months",
        },inplace=True)
        
        category_types_features = pd.merge(category_types_features, items_sold_in_month, on="item_category_type", how="left")

    category_types_features.to_csv("generated/" + features_folder_name + "features_category_types.csv", index=False)'''

    #---------------------------------------------------------------------------------------
    
    '''print("Features of shops")

    shops_features = df_aspects[["shop_id"]].drop_duplicates(subset=["shop_id"])

    ## Stats last n months.

    months = [month_to_predict - 1 - i for i in range(3)]

    for month in months:
        items_sold_in_month = df_ts[df_ts["date_block_num"] == month].groupby("shop_id").agg({"item_cnt_month":"mean"}).reset_index()
        
        items_sold_in_month.rename(columns={
            "item_cnt_month":"shop_item_mean_shifted_" + str(month_to_predict - month) + "_months",
        }, inplace=True)

        shops_features = pd.merge(shops_features, items_sold_in_month, on="shop_id", how="left")


    shops_features.to_csv("generated/" + features_folder_name +"features_shops.csv", index=False)'''

    #---------------------------------------------------------------------------------------

    print("Features of items and shops")

    features_items_and_shops = df_aspects[["item_id", "shop_id"]].drop_duplicates(subset=["item_id", "shop_id"])

    #### Items sold in shop last month

    item_purchases_by_shop_last_month = df_ts[df_ts["date_block_num"] == max_train_month].groupby(["shop_id", "item_id"]).agg({"item_cnt_month":"sum"}).reset_index().rename(columns={"item_cnt_month":"purchases_item_shop_last_month"})
    features_items_and_shops = pd.merge(features_items_and_shops, item_purchases_by_shop_last_month, on=["item_id", "shop_id"], how="left")


    #### Item sold in shop in last n months

    months = [max_train_month - 1 - i for i in range(2)]

    for month in months:
        item_purchases_by_shop_in_month = df_ts[df_ts["date_block_num"] == month].groupby(["shop_id", "item_id"]).agg({"item_cnt_month":"sum"}).reset_index().rename(columns={"item_cnt_month":"purchases_item_in_shop_month_" + str(month_to_predict - month)})
        features_items_and_shops = pd.merge(features_items_and_shops, item_purchases_by_shop_in_month, on=["shop_id", "item_id"], how="left")

    #### Last purchase

    with_last_purchase_month = df.groupby(["item_id", "shop_id"]).agg({"date_block_num":"max"}).reset_index().rename(columns={"date_block_num":"month_last_purchase_of_item_in_shop"})
    
    features_items_and_shops = pd.merge(features_items_and_shops, with_last_purchase_month, on=["shop_id", "item_id"], how="left")

    features_items_and_shops["labeled_month"] = month_to_predict

    features_items_and_shops["last_sale_shop_item"] = features_items_and_shops["labeled_month"] - features_items_and_shops["month_last_purchase_of_item_in_shop"]

    features_items_and_shops.drop(columns=["labeled_month"], inplace=True)

    ## Historical sales for shop in specific item (sum, mean, std, min, max)
    
    '''sales_of_item_in_shop_by_month_h = df_ts.groupby(["item_id", "shop_id"]).agg({"item_cnt_month": ["mean", "sum", "std", "max", "min"]}).reset_index().rename(columns={"sales_in_month":"average_sales_for_item_and_shop_by_month"})

    sales_of_item_in_shop_by_month_h.columns = ['_'.join(col).strip() for col in sales_of_item_in_shop_by_month_h.columns.values]

    sales_of_item_in_shop_by_month_h.rename(columns={
    "shop_id_":"shop_id",
        "item_id_":"item_id",
        "item_cnt_month_sum":"hist_sales_sum_item_by_shop",
        "item_cnt_month_mean":"hist_sales_mean_item_by_shop",
        "item_cnt_month_std":"hist_sales_std_item_by_shop",
        "item_cnt_month_min":"hist_sales_min_item_by_shop",
        "item_cnt_month_max":"hist_sales_max_item_by_shop"
    }, inplace=True)

    features_items_and_shops = pd.merge(features_items_and_shops, sales_of_item_in_shop_by_month_h, on=["shop_id", "item_id"],how="left")'''


    features_items_and_shops.to_csv("generated/" + features_folder_name + "features_items_and_shop.csv", index=False)    

    #--------------------------------------------------------------------------------------------

    print("Features of items")

    ## Stats last n months

    items_features = df_aspects[["item_id"]].drop_duplicates(subset=["item_id"])

    months = [month_to_predict - i for i in [1,2,3,6]]

    for month in months:
        items_sold_in_month = df_ts[df_ts["date_block_num"] == month].groupby("item_id").agg({"item_cnt_month":["mean", "sum", "max", "std"]}).reset_index()
        items_sold_in_month.columns = ['_'.join(col).strip() for col in items_sold_in_month.columns.values]
        
        items_sold_in_month.rename(columns={
            "item_id_":"item_id",
            "item_cnt_month_sum":"item_purchases_shifted_" + str(month_to_predict - month) + "_months",
            "item_cnt_month_mean":"item_purchases_mean_shifted_" + str(month_to_predict - month) + "_months",
            "item_cnt_month_max":"item_purchases_max_shifted_" + str(month_to_predict - month) + "_months",
            "item_cnt_month_std":"item_purchases_std_shifted_" + str(month_to_predict - month) + "_months",
        },
        inplace=True)
        items_features = pd.merge(items_features, items_sold_in_month, on="item_id", how="left")

    ## Rolling features

    items_sells_historically = df_ts[df_ts.item_id.isin(items_features.item_id)].groupby(["item_id", "date_block_num"]).agg({"item_cnt_month":"sum"}).reset_index()
    grouped = items_sells_historically.sort_values(by="date_block_num").groupby("item_id").agg({"item_cnt_month":[rolling_mean, rolling_std, rolling_max, rolling_min, diffmean]}).reset_index()
    grouped.columns = ['_'.join(col).strip() + "_item" for col in grouped.columns.values]
    grouped.rename(columns={"item_id__item":"item_id"}, inplace=True)

    items_features = pd.merge(items_features, grouped, on="item_id", how="left")

    ## Prices features

    '''df_prices = df_aux.copy()

    with_price_features = df_prices.groupby("item_id").agg({"item_price":["max", "mean", "min", "std"]}).reset_index()
    with_price_features.columns = ['_'.join(col).strip() for col in with_price_features.columns.values]
    with_price_features["diff_price_max_min"] = with_price_features["item_price_max"] - with_price_features["item_price_min"]
    with_price_features.rename(columns={"item_id_":"item_id"}, inplace=True)

    items_features = pd.merge(items_features, with_price_features, on="item_id", how="left")'''


    '''##Total shops with item

    with_different_shops = df_aux.groupby("item_id").agg({"shop_id":"nunique"}).reset_index().rename(columns={"shop_id":"total_shops_item_is_sell"})
    items_features = pd.merge(items_features, with_different_shops, on="item_id", how="left")
    items_features["total_shops_item_is_sell"].fillna(0)

    ## Total months item is sell

    months_of_sales_by_item = df.groupby("item_id").agg({"date_block_num":"nunique"}).reset_index().rename(columns={"date_block_num":"months_item_has_sales"})
    items_features = pd.merge(items_features, months_of_sales_by_item, on="item_id", how="left")'''

   
    items_features.to_csv("generated/" + features_folder_name + "features_items.csv", index=False)
