from gdeltdoc import GdeltDoc, Filters

gd = GdeltDoc()
f = Filters(keyword="Israel terrorism",
            start_date="2025-05-20",
            end_date="2025-05-25",
            num_records=10)

articles = gd.article_search(f)
print(articles.head())
