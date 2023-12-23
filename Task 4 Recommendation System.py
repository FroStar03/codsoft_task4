import pandas as pd
from sklearn.metrics.pairwise
import cosine_similarity

data = {
    'User': ['User1', 'User2', 'User3', 'User4', 'User5'],
    'Book1': [5, 4, 0, 0, 3],
    'Book2': [4, 0, 5, 4, 2],
    'Book3': [0, 2, 4, 5, 0],
    'Book4': [3, 4, 0, 0, 5],
    'Book5': [0, 0, 3, 4, 4]
}

df = pd.DataFrame(data)

def get_recommendations(user, df):
    user_ratings = df[df['User'] == user].iloc[:, 1:]
    similar_users = df[df['User'] != user].iloc[:, 1:]

    similarities = cosine_similarity(user_ratings, similar_users)[0]
    
    similar_user_index = similarities.argmax()

    recommendations = similar_users.iloc[similar_user_index]
    recommended_books = recommendations[recommendations.apply(lambda x: x > 0)].index.tolist()

    return recommended_books

user_recommendations = get_recommendations('User1', df)
print(f"Recommended books for User1: {user_recommendations}")
