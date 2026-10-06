"""Synthetic online-store dataset with known preference, bundle, and season effects.

Latent user tastes are stored in users.csv for diagnostics only. Models must not
read that file: they see products and interactions, the same inputs a shop has.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
SEED = 42
N_USERS = 500
N_COLD = 40
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-09-30")
NEW_ITEM_LAUNCH = pd.Timestamp("2026-08-01")


def _sku(
    category: str,
    brand: str,
    title: str,
    price: float,
    tags: str,
    season: str = "all",
    bundle: str = "",
    sensitive: int = 0,
    is_new: bool = False,
) -> dict:
    return {
        "title": title,
        "category": category,
        "brand": brand,
        "price": float(price),
        "season": season,
        "tags": tags,
        "description": (
            f"{title}. Категория {category}, бренд {brand}. {tags}. Сезон: {season}."
        ),
        "bundle": bundle,
        "sensitive": int(sensitive),
        "launch_date": NEW_ITEM_LAUNCH.date().isoformat() if is_new else "",
    }


def build_catalog() -> pd.DataFrame:
    rows: list[dict] = []
    add = rows.append

    for brand, title, price in [
        ("Nova", "Смартфон Nova Note 12", 19990),
        ("Nova", "Смартфон Nova Note 13", 24990),
        ("Orbit", "Смартфон Orbit A35", 32990),
        ("Orbit", "Смартфон Orbit S23", 69990),
        ("Lumen", "Смартфон Lumen X10", 39990),
    ]:
        add(_sku("electronics", brand, title, price, "smartphone phone мобильный связь", bundle="phone"))

    for brand, title, price in [
        ("Nova", "Чехол Nova силиконовый", 990),
        ("Nova", "Чехол Nova прозрачный", 790),
        ("Orbit", "Чехол Orbit книжка", 1490),
        ("Lumen", "Чехол Lumen противоударный", 1290),
    ]:
        add(_sku("electronics", brand, title, price, "case чехол phone аксессуар", bundle="phone"))

    for brand, title, price in [
        ("Nova", "Зарядка Nova 33W", 1490),
        ("Orbit", "Зарядка Orbit 65W", 2490),
        ("Lumen", "Кабель Lumen USB-C", 690),
    ]:
        add(_sku("electronics", brand, title, price, "charger зарядка cable phone", bundle="phone"))

    for brand, title, price in [
        ("Nova", "Наушники Nova Buds", 2990),
        ("Orbit", "Наушники Orbit Buds", 4490),
        ("Lumen", "Наушники Lumen Air", 5990),
        ("Lumen", "Наушники Lumen Studio Max", 8990),
    ]:
        add(_sku("electronics", brand, title, price, "earbuds headphones аудио музыка"))

    add(_sku("electronics", "Lumen", "Наушники Lumen Studio", 11990, "headphones аудио музыка студийные", is_new=True))
    add(_sku("electronics", "Nova", "Внешний аккумулятор Nova 10000", 1990, "powerbank аккумулятор phone", bundle="phone"))
    add(_sku("electronics", "Orbit", "Внешний аккумулятор Orbit 20000", 3490, "powerbank аккумулятор travel"))
    add(_sku("electronics", "Lumen", "Телевизор Lumen 43", 34990, "tv телевизор экран дом"))
    add(_sku("electronics", "Orbit", "Телевизор Orbit 55", 54990, "tv телевизор экран дом"))
    add(_sku("electronics", "Lumen", "Колонка Lumen Mini", 3990, "speaker колонка аудио музыка"))
    add(_sku("electronics", "Nova", "Колонка Nova Home", 5990, "speaker колонка аудио дом", is_new=True))

    for title, price in [
        ("Обогреватель Teplo 1500", 4590),
        ("Обогреватель Teplo керамический", 6290),
        ("Конвектор Teplo Compact", 3990),
    ]:
        add(_sku("home", "Teplo", title, price, "heater обогреватель тепло зима", season="winter"))

    add(_sku("home", "Domov", "Вентилятор Domov напольный", 3290, "fan вентилятор прохлада лето", season="summer"))
    add(_sku("home", "Domov", "Вентилятор Domov настольный", 1890, "fan вентилятор прохлада лето", season="summer"))
    add(_sku("home", "Keram", "Лампа Keram настольная", 2490, "lamp лампа свет дом"))
    add(_sku("home", "Keram", "Торшер Keram", 5990, "lamp торшер свет дом"))
    add(_sku("home", "Keram", "Гирлянда Keram тёплый свет", 990, "light гирлянда уют дом", season="winter", is_new=True))
    add(_sku("home", "Domov", "Постельное бельё Domov хлопок", 3490, "bedding постель текстиль дом"))
    add(_sku("home", "Domov", "Подушка Domov", 1490, "bedding подушка сон дом"))
    add(_sku("home", "Domov", "Одеяло Domov зима", 3990, "blanket одеяло тепло сон", season="winter"))
    add(_sku("home", "Keram", "Пылесос Keram вертикальный", 12990, "vacuum пылесос уборка дом"))
    add(_sku("home", "Teplo", "Чайник Teplo 1.7", 2490, "kettle чайник кухня"))
    add(_sku("home", "Keram", "Кофеварка Keram", 8990, "coffee кофе кухня"))

    for brand, title, price in [
        ("Runly", "Кроссовки Runly Pace", 7990),
        ("Runly", "Кроссовки Runly City", 8990),
        ("Vertex", "Кроссовки Vertex Trail", 10990),
        ("Alta", "Кроссовки Alta Daily", 6990),
    ]:
        add(_sku("sports", brand, title, price, "shoes кроссовки бег спорт", bundle="run"))

    add(_sku("sports", "Runly", "Носки Runly беговые 3 пары", 990, "socks носки бег", bundle="run"))
    add(_sku("sports", "Vertex", "Носки Vertex спортивные", 790, "socks носки спорт", bundle="run"))
    add(_sku("sports", "Alta", "Лыжи Alta Classic", 18990, "ski лыжи зима снег", season="winter"))
    add(_sku("sports", "Alta", "Ботинки лыжные Alta", 12990, "ski ботинки зима", season="winter"))
    add(_sku("sports", "Runly", "Купальник Runly", 3490, "swim купальник лето бассейн", season="summer"))
    add(_sku("sports", "Vertex", "Очки для плавания Vertex", 1490, "swim очки лето бассейн", season="summer"))
    add(_sku("sports", "Vertex", "Гантели Vertex 2x5 кг", 2990, "fitness гантели тренировка"))
    add(_sku("sports", "Runly", "Коврик для йоги Runly", 1990, "fitness коврик йога"))
    add(_sku("sports", "Alta", "Мяч футбольный Alta", 2490, "ball мяч футбол"))
    add(_sku("sports", "Vertex", "Бутылка для воды Vertex", 890, "bottle бутылка спорт", is_new=True))

    for brand, title, price in [
        ("Flora", "Шампунь Flora объём", 690),
        ("Nive", "Шампунь Nive восстановление", 590),
        ("Purelab", "Шампунь Purelab мягкий", 750),
    ]:
        add(_sku("beauty", brand, title, price, "shampoo шампунь волосы уход", bundle="hair"))

    for brand, title, price in [
        ("Flora", "Бальзам Flora объём", 640),
        ("Nive", "Бальзам Nive восстановление", 560),
        ("Purelab", "Бальзам Purelab мягкий", 720),
    ]:
        add(_sku("beauty", brand, title, price, "conditioner бальзам волосы уход", bundle="hair"))

    add(_sku("beauty", "Flora", "Солнцезащитный крем Flora SPF50", 990, "sunscreen спф лето защита", season="summer"))
    add(_sku("beauty", "Purelab", "Солнцезащитный стик Purelab", 890, "sunscreen спф лето лицо", season="summer"))
    add(_sku("beauty", "Nive", "Крем для рук Nive зима", 390, "cream крем руки зима", season="winter"))
    add(_sku("beauty", "Flora", "Крем для лица Flora питание", 1290, "cream крем лицо зима уход", season="winter"))
    add(_sku("beauty", "Purelab", "Сыворотка Purelab", 2490, "serum сыворотка лицо уход", is_new=True))
    add(_sku("beauty", "Flora", "Туалетная вода Flora", 3990, "perfume парфюм аромат"))
    add(_sku("beauty", "Nive", "Дезодорант Nive", 290, "deodorant дезодорант уход"))

    add(_sku("health", "Purelab", "Фолиевая кислота Purelab", 490, "vitamin фолиевая витамины здоровье", bundle="private"))
    add(_sku("health", "Purelab", "Витамин D Purelab", 540, "vitamin витамин здоровье"))
    add(_sku("health", "Nive", "Витамин C Nive", 390, "vitamin витамин иммунитет"))
    add(_sku(
        "health",
        "Flora",
        "Тест на беременность Flora",
        250,
        "test тест личное здоровье",
        bundle="private",
        sensitive=1,
    ))
    add(_sku("health", "Nive", "Пластырь Nive набор", 180, "bandage пластырь аптечка"))

    for brand, title, price, tags in [
        ("Listok", "Книга: короткая история времени", 890, "book книга наука нонфикшн"),
        ("More", "Книга: город и море", 750, "book книга роман проза"),
        ("Arka", "Книга: рецепты будней", 990, "book книга кулинария дом"),
        ("Listok", "Книга: введение в алгоритмы", 1490, "book книга программирование учёба"),
        ("More", "Книга: детский атлас", 690, "book книга дети атлас"),
        ("Arka", "Книга: бег без травм", 820, "book книга спорт бег"),
        ("Listok", "Книга: зимние рассказы", 640, "book книга рассказы зима", ),
        ("More", "Книга: новый роман осени", 880, "book книга роман новинка"),
    ]:
        is_new = "новинка" in tags
        add(_sku("books", brand, title, price, tags, season="all", is_new=is_new))

    for brand, title, price in [
        ("Forma", "Пальто Forma шерсть", 14990),
        ("Thread", "Пуховик Thread", 12990),
        ("Northwind", "Пальто Northwind", 16990),
    ]:
        add(_sku("fashion", brand, title, price, "coat пальто верхняя одежда зима", season="winter"))

    for brand, title, price in [
        ("Forma", "Платье Forma лён", 4990),
        ("Thread", "Сарафан Thread", 3990),
        ("Northwind", "Платье Northwind", 6990),
    ]:
        add(_sku("fashion", brand, title, price, "dress платье лето одежда", season="summer"))

    add(_sku("fashion", "Forma", "Джинсы Forma прямые", 5490, "jeans джинсы одежда"))
    add(_sku("fashion", "Thread", "Джинсы Thread", 4590, "jeans джинсы одежда"))
    add(_sku("fashion", "Northwind", "Рубашка Northwind", 3990, "shirt рубашка одежда"))
    add(_sku("fashion", "Forma", "Футболка Forma базовая", 1990, "tshirt футболка одежда"))
    add(_sku("fashion", "Thread", "Футболка Thread", 1490, "tshirt футболка лето одежда", season="summer"))
    add(_sku("fashion", "Northwind", "Шарф Northwind шерсть", 2490, "scarf шарф зима аксессуар", season="winter"))
    add(_sku("fashion", "Forma", "Шапка Forma", 1990, "hat шапка зима", season="winter"))
    add(_sku("fashion", "Thread", "Кроссовки Thread Street", 6490, "sneakers обувь город", is_new=True))

    frame = pd.DataFrame(rows)
    frame.insert(0, "item_id", np.arange(1, len(frame) + 1))
    return frame


def _season(ts: pd.Timestamp) -> str:
    month = int(ts.month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    products = build_catalog()
    categories = sorted(products["category"].unique())
    launch = pd.to_datetime(products["launch_date"].replace("", pd.NA))
    n_items = len(products)
    day_span = (END - START).days

    item_pos = {int(i): pos for pos, i in enumerate(products["item_id"])}
    bundles: dict[str, list[int]] = {}
    for row in products.itertuples(index=False):
        if row.bundle:
            bundles.setdefault(row.bundle, []).append(int(row.item_id))

    users: list[dict] = []
    events: list[dict] = []

    def sample_item(fav_cats: set[str], fav_brands: set[str], ts: pd.Timestamp) -> int:
        season = _season(ts)
        weights = np.ones(n_items, dtype=float)
        for pos, prod in enumerate(products.itertuples(index=False)):
            if prod.category in fav_cats:
                weights[pos] *= 4.0
            if prod.brand in fav_brands:
                weights[pos] *= 2.2
            if int(prod.sensitive) == 1:
                weights[pos] *= 0.02
            if prod.season == season:
                weights[pos] *= 3.0
            elif prod.season != "all":
                weights[pos] *= 0.25
            if pd.notna(launch.iloc[pos]) and ts < launch.iloc[pos]:
                weights[pos] = 0.0
        total = weights.sum()
        if total <= 0:
            weights = np.ones(n_items, dtype=float)
            total = weights.sum()
        return int(products.iloc[int(rng.choice(n_items, p=weights / total))]["item_id"])

    def push(user_id: int, item_id: int, event_type: str, ts: pd.Timestamp, rating: float | None) -> None:
        events.append(
            {
                "user_id": user_id,
                "item_id": item_id,
                "event_type": event_type,
                "rating": rating if rating is not None else np.nan,
                "timestamp": ts,
            }
        )

    for user_id in range(1, N_USERS + 1):
        segment = "cold" if user_id <= N_COLD else "warm"
        pref_categories = [category for category in categories if category != "health"]
        fav_cats = set(rng.choice(pref_categories, size=2, replace=False))
        brand_pool = products.loc[products["category"].isin(fav_cats), "brand"].unique()
        fav_brands = set(rng.choice(brand_pool, size=min(2, len(brand_pool)), replace=False))
        users.append(
            {
                "user_id": user_id,
                "segment": segment,
                "fav_categories": "|".join(sorted(fav_cats)),
                "fav_brands": "|".join(sorted(fav_brands)),
            }
        )

        n_events = 3 if segment == "cold" else int(rng.integers(12, 34))
        offsets = np.sort(rng.integers(0, day_span, size=n_events))
        minutes = rng.integers(0, 24 * 60, size=n_events)
        owned: set[int] = set()

        for step, (offset, minute) in enumerate(zip(offsets, minutes)):
            ts = START + pd.Timedelta(days=int(offset), minutes=int(minute))
            item_id = sample_item(fav_cats, fav_brands, ts)
            last = step == n_events - 1
            if segment == "cold" and last:
                event_type = "purchase"
            elif rng.random() < 0.42:
                event_type = "view"
            elif rng.random() < 0.35:
                event_type = "cart"
            else:
                event_type = "purchase"

            rating = None
            if event_type == "purchase":
                prod = products.iloc[item_pos[item_id]]
                match = prod.category in fav_cats and prod.brand in fav_brands
                raw = (4.6 if match else 3.3) + float(rng.normal(0, 0.45))
                rating = float(min(5, max(1, round(raw))))
                owned.add(item_id)
            push(user_id, item_id, event_type, ts, rating)

            if event_type == "purchase" and segment != "cold":
                group = products.iloc[item_pos[item_id]]["bundle"]
                if group and rng.random() < 0.7:
                    options = [i for i in bundles[group] if i not in owned]
                    if options:
                        extra = int(rng.choice(options))
                        extra_rating = float(min(5, max(1, round(4.2 + rng.normal(0, 0.4)))))
                        push(
                            user_id,
                            extra,
                            "purchase",
                            ts + pd.Timedelta(hours=int(rng.integers(1, 36))),
                            extra_rating,
                        )
                        owned.add(extra)

    # A stable co-purchase of a private item with vitamins, so the ethical
    # example does not depend on a rare random draw.
    private_ids = bundles["private"]
    vitamin_id = int(
        products.loc[products["title"].str.contains("Фолиевая"), "item_id"].iloc[0]
    )
    test_id = int(
        products.loc[products["title"].str.contains("беременность"), "item_id"].iloc[0]
    )
    for user_id in range(80, 115):
        base = START + pd.Timedelta(days=int(rng.integers(40, 500)))
        push(user_id, vitamin_id, "purchase", base, 5.0)
        push(user_id, test_id, "purchase", base + pd.Timedelta(days=1), 5.0)
        _ = private_ids

    interactions = pd.DataFrame(events).sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    users_frame = pd.DataFrame(users)
    return products, users_frame, interactions


def main() -> None:
    products, users, interactions = generate()
    products.to_csv(OUT_DIR / "products.csv", index=False, encoding="utf-8")
    users.to_csv(OUT_DIR / "users.csv", index=False, encoding="utf-8")
    interactions.to_csv(OUT_DIR / "interactions.csv", index=False, encoding="utf-8")

    purchases = interactions[interactions["event_type"] == "purchase"]
    density = purchases.groupby(["user_id", "item_id"]).ngroups / (len(users) * len(products))
    cold = int((users["segment"] == "cold").sum())
    print(f"products: {len(products)}")
    print(f"users: {len(users)} (cold {cold})")
    print(f"interactions: {len(interactions)}")
    print(f"purchases: {len(purchases)}")
    print(f"purchase density: {density:.4f}")
    print(f"sensitive items: {int(products['sensitive'].sum())}")


if __name__ == "__main__":
    main()
