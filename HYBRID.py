

#### HYBRID RECOMMENDER SYSTEM ####

### İŞ PROBLEMİ ###

# ID'si verilen kullanıcı için item-based ve user-based recommender yöntemlerini
# kullanarak tahmin yapınız.

### VERİ SETİ HİKAYESİ ###

# Veri seti, bir film tavsiye hizmeti olan MovieLens tarafından sağlanmıştır.
# İçerisinde filmler ile birlikte bu filmlere yapılan derecelendirme puanlarını
# barındırmaktadır.
# 27.278 filmde 2.000.0263 derecelendirme içermektedir.
# Bu veriler 138.493 kullanıcı tarafından 09 Ocak 1995 ile 31 Mart 2015
# tarihleri arasında oluşturulmuştur. Bu veri seti ise 17 Ekim 2016 tarihinde
# oluşturulmuştur.
# Kullanıcılar rastgele seçilmiştir. Seçilen tüm kullanıcıların en az 20 filme oy
# verdiği bilgisi mevcuttur.

# GÖREV-1: Veri setinin hazırlanması

import pandas as pd
pd.set_option('display.max_columns', 20)

def create_user_movie_df():
    movie = pd.read_csv('datasets/movie.csv')
    rating = pd.read_csv('datasets/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df=create_user_movie_df()
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# GÖREV-2: Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.

random_user_df= user_movie_df[user_movie_df.index== random_user]
movies_watched=random_user_df.columns[random_user_df.notna().any()].tolist()
len(movies_watched)
movies_watched
## İzlediği filmler
# ['Ace Ventura: Pet Detective (1994)',
#  'Ace Ventura: When Nature Calls (1995)',
#  'Aladdin (1992)',
#  'American President, The (1995)',
#  'Apollo 13 (1995)',
#  'Babe (1995)',
#  'Bullets Over Broadway (1994)',
#  'Clueless (1995)',
#  'Disclosure (1994)',
#  'Forrest Gump (1994)',
#  'Four Weddings and a Funeral (1994)',
#  'Home Alone (1990)',
#  'Jurassic Park (1993)',
#  'Like Water for Chocolate (Como agua para chocolate) (1992)',
#  'Little Women (1994)',
#  "Mr. Holland's Opus (1995)",
#  'Mrs. Doubtfire (1993)',
#  'Much Ado About Nothing (1993)',
#  "Muriel's Wedding (1994)",
#  'Nine Months (1995)',
#  'Operation Dumbo Drop (1995)',
#  'Piano, The (1993)',
#  'Postman, The (Postino, Il) (1994)',
#  'Ready to Wear (Pret-A-Porter) (1994)',
#  'Remains of the Day, The (1993)',
#  'Sabrina (1995)',
#  "Schindler's List (1993)",
#  'Secret Garden, The (1993)',
#  'Sense and Sensibility (1995)',
#  'Shadowlands (1993)',
#  'Silence of the Lambs, The (1991)',
#  'Star Trek: Generations (1994)',
#  'Stargate (1994)']

# GÖREV-3: Aynı filmi izleyen diğer kullanıcıların verisine ve Idlerine erişiniz.

movies_watched_df= user_movie_df[movies_watched]
movies_watched_df.head()

user_movie_count=movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

users_same_movies.head()

# GÖREV-4 : Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz.

final_df=pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                    random_user_df[movies_watched]])
final_df.T.corr()

corr_df=final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

rating = pd.read_csv('datasets/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings.head(50)

# Random userı çıkardık.
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# GÖREV-5: Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz.

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

top_users_ratings.groupby("movieId").agg({"weighted_rating":"mean"})

recommedation_df=top_users_ratings.groupby("movieId").agg({"weighted_rating":"mean"})
recommedation_df=recommedation_df.reset_index()

movies_to_be_recommend=recommedation_df.sort_values("weighted_rating" ,ascending=False).iloc[0:5,0:5]

movie= pd.read_csv("datasets/movie.csv")
movies_to_be_recommend.merge(movie[["movieId","title"]])

# GÖREV-6: Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
# ▪ 5 öneri user-based
# ▪ 5 öneri item-based
# olacak şekilde 10 öneri yapınız.

user=108170
movie= pd.read_csv("datasets/movie.csv")
rating=pd.read_csv("datasets/rating.csv")

#Item-Based Öneri

movie_id=rating[(rating["userId"]==user) & (rating["rating"]==5)].\
             sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

movie_name=user_movie_df[movie[movie.movieId==movie_id].title.values[0]]

movies_from_item_based=user_movie_df.corrwith(movie_name).sort_values(ascending=False)

movies_from_item_based[1:6].index

#User-Based Öneri

movies_from_used_based = movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"]

movies_from_used_based.head(5)
