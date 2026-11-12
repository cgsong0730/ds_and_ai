import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

df = sns.load_dataset("mpg")
df = df.dropna()

features = ["cylinders", "displacement", "horsepower",
            "weight", "acceleration", "model_year"]
X = df[features]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 연비 데이터의 X_train, X_test, y_train, y_test 사용 (특성 6개)
for d in [1, 2, 3, 5, 8, None]:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    print(d, round(tree.score(X_train, y_train), 3), round(tree.score(X_test, y_test), 3), tree.get_n_leaves())