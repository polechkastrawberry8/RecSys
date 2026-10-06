"""Qualitative look at one warm user, one cold user, and one product page.

Models are fit on each user's past only. A mark means the product was purchased
later, in the held-out part of the log. Favorite categories printed below come
from the data generator. The models never read that file.
"""

from __future__ import annotations

import pandas as pd

from recsys.data import load_dataset, season_of, time_split
from recsys.service import Storefront


def _show(title: str, recs: pd.DataFrame, products: pd.DataFrame, hits: set[int] | None = None) -> None:
    print(f"\n{title}")
    if recs.empty:
        print("  (пусто)")
        return
    meta = products.set_index("item_id")
    hits = hits or set()
    for row in recs.itertuples(index=False):
        info = meta.loc[int(row.item_id)]
        mark = "  ← куплено в тесте" if int(row.item_id) in hits else ""
        print(
            f"  {info.title}  |  {info.category}, {info.brand}, {info.price:.0f} ₽"
            f"  |  {row.reason}{mark}"
        )


def _hits(test: pd.DataFrame, user_id: int) -> set[int]:
    rows = test[(test["user_id"] == user_id) & (test["event_type"] == "purchase")]
    return {int(item) for item in rows["item_id"]}


def main() -> None:
    products, users, interactions = load_dataset()
    train, test = time_split(interactions)
    store = Storefront().fit(train, products)

    relevant = test[test["event_type"].eq("purchase")].groupby("user_id").size()
    train_size = train.groupby("user_id").size()
    warm_ids = [
        int(user)
        for user in relevant.index
        if int(train_size.get(user, 0)) >= 8 and int(relevant.loc[user]) >= 1
    ]
    warm_id = warm_ids[0]
    profile = users.loc[users["user_id"] == warm_id].iloc[0]
    request_season = season_of(test.loc[test["user_id"] == warm_id, "timestamp"].min())
    print(
        f"Тёплый пользователь {warm_id}. Скрытый профиль генератора "
        f"(модель его не видит): {profile.fav_categories}; бренды {profile.fav_brands}."
    )
    print(f"Сезон запроса, по времени первого тестового события: {request_season}.")
    home = store.home(warm_id, k=5, season=request_season)
    hits = _hits(test, warm_id)
    for shelf, frame in home.items():
        _show(shelf, frame, products, hits)

    cold_id = int(users.loc[users["segment"].eq("cold"), "user_id"].iloc[0])
    cold_profile = users.loc[users["user_id"] == cold_id].iloc[0]
    print(
        f"\nХолодный пользователь {cold_id}. Событий в обучении: "
        f"{int(train_size.get(cold_id, 0))}. Скрытые категории: {cold_profile.fav_categories}."
    )
    cold_home = store.home(cold_id, k=5)
    cold_hits = _hits(test, cold_id)
    for shelf in ("user_cf", "content", "popular", "for_you"):
        _show(shelf, cold_home[shelf], products, cold_hits)

    phones = products[products["tags"].str.contains("smartphone", na=False)]
    phone_buys = train[
        train["item_id"].isin(phones["item_id"]) & train["event_type"].eq("purchase")
    ]
    phone_id = int(phone_buys.groupby("item_id").size().idxmax())
    phone = products.loc[products["item_id"] == phone_id].iloc[0]
    print(f"\nКарточка товара: {phone.title} ({phone.brand}, {phone.price:.0f} ₽)")
    for shelf, frame in store.product_page(phone_id, k=5, user_id=warm_id).items():
        _show(shelf, frame, products)

    vitamin = products[products["title"].str.contains("Фолиевая", na=False)].iloc[0]
    print(f"\nС этим также покупают, без фильтра, для «{vitamin.title}»:")
    _show(
        "also_bought_unfiltered",
        store.also_bought_unfiltered.recommend(k=5, context={"item_id": int(vitamin.item_id)}),
        products,
    )
    print(f"Та же полка с фильтром чувствительных товаров:")
    _show(
        "also_bought_filtered",
        store.also_bought.recommend(k=5, context={"item_id": int(vitamin.item_id)}),
        products,
    )


if __name__ == "__main__":
    main()
