# __Homework 8:__ Natural Language Processing


_Name_: Casey Chadwell  
_SID_: 3033291861   
_Class_: IEOR 135   
_GitHub_:    

---

__Preliminaries__: For this assignment, I began by setting the parameter `random_state` to `42` in the random forest classifier used in the `train_predict_sentiment` function. This was mentioned on Piazza as a good idea to ensure the results I obtain are replicable. The changed line is as follows:

`forest = RandomForestClassifier(n_estimators = 50, random_state=42)`

For each part, I changed the function name `original_clean_reviews` to something more appropriate for the task at hand for the sake of clarity. (e.g. `lemmatized_clean_reviews`)

---

## 1. __Unigram Setting__
For original reviews with unigram and 1000 max features, the functions were defined as follows:

```
original_clean_reviews=review_cleaner(
    train['review'],
    lemmatize = False,
    stem = False
)

train_predict_sentiment(
    cleaned_reviews = original_clean_reviews,
    y = train["sentiment"],
    ngram = 1,
    max_features = 1000
)
```

The original cleaning settings produced the following accuracies from the random forest classifier:

> The training accuracy is:  1.0   
> The validation accuracy is:  0.8166

#### __A. lemmatized reviews__:

To create lemmatized reviews, the parameter `lemmatize` in the function `original_clean_reviews` was set to `True` and the parameter `stem` was left as `False`. The function calls were as follows:

```
lemmatized_clean_reviews = review_cleaner(
    train['review'],
    lemmatize = True,
    stem = False
)

train_predict_sentiment(
    cleaned_reviews = lemmatized_clean_reviews,
    y = train["sentiment"],
    ngram = 1,
    max_features = 1000
)
```
Lemmatizing the reviews produced the following accuracies:

> The training accuracy is:  1.0   
> The validation accuracy is:  0.8216

__Brief Overview__:
Lemmatizing the reviews resulted in a 0.005 increase in validation accuracy while maintaining a perfect 100% training accuracy.

#### __B. Stemmed Reviews__:

To create stemmed reviews, the parameter `lemmatize` in the function `original_clean_reviews` was set to `False` and the parameter `stem` was set to `True`. The function calls were as follows:

```
stemmed_clean_reviews = review_cleaner(
    train['review'],
    lemmatize = False,
    stem = True
)

train_predict_sentiment(
    cleaned_reviews = stemmed_clean_reviews,
    y = train["sentiment"],
    ngram = 1,
    max_features = 1000
)
```
Stemming the reviews produced the following accuracies:

> The training accuracy is:  1.0    
> The validation accuracy is:  0.8196

__Brief Overview__:
Stemming the reviews resulted in a 0.003 increase in validation accuracy while also maintaining a perfect training accuracy.

---

## 2. __Bigram Setting__

For this part, the results were obtained by changing the parameter `ngram` from `1` to `2` in the `train_predict_sentiment` function.

For original reviews with unigram and 1000 max features, the functions were defined as follows:

```
original_clean_reviews=review_cleaner(
    train['review'],
    lemmatize = False,
    stem = False
)

train_predict_sentiment(
    cleaned_reviews = original_clean_reviews,
    y = train["sentiment"],
    ngram = 2,
    max_features = 1000
)
```

The original cleaning settings produced the following accuracies from the random forest classifier:

> The training accuracy is:  1.0   
> The validation accuracy is:  0.8166

#### __A. lemmatized reviews__:

In the call to `original_clean_reviews`, `lemmatized` was set to `True` and `stem` was set to `False`. The parameter `ngram` was changed to `2`. The function calls were as follows:

```
lemmatized_2_clean_reviews = review_cleaner(
    train['review'],
    lemmatize = True,
    stem = False
)

train_predict_sentiment(
    cleaned_reviews = lemmatized_2_clean_reviews,
    y = train["sentiment"],
    ngram = 2,
    max_features = 1000
)
```
Lemmatizing the reviews and increasing `ngram` to 2 produced the following accuracies:

> The training accuracy is:  1.0   
> The validation accuracy is:  0.8196

__Brief Overview__:
Increasing `ngram` to 2 produced a slight 0.003 increase in validation accuracy compared to the original cleaning function, however the validation accuracy is lower than with `ngram = 1` for lemmatized reviews.

#### __B. Stemmed Reviews__:

In the call to `original_clean_reviews`, `lemmatized` was set to `False` and `stem` was set to `True`. The parameter `ngram` was set to `2`. The function calls were as follows:

```
stemmed_2_clean_reviews = review_cleaner(
    train['review'],
    lemmatize = False,
    stem = True
)

train_predict_sentiment(
    cleaned_reviews = stemmed_2_clean_reviews,
    y = train["sentiment"],
    ngram = 2,
    max_features = 1000
)
```
Lemmatizing the reviews and increasing `ngram` to 2 produced the following accuracies:

> The training accuracy is:  1.0    
> The validation accuracy is:  0.8196

__Brief Overview__:
Stemming with `ngram = 2` produced the same validation accuracy as stemming with `ngram = 1`.

---

## 3. __Max Features__: with Unigram Setting

For this part, the results were obtained by using lemmatized reviews with `ngram=1`. In the following parts, the review cleaning function call was left as:

```
lemmatized_clean_reviews = review_cleaner(
    train['review'],
    lemmatize = True,
    stem = False
)
```

In the `train_predict_sentiment` function, the value of `max_features` was changed to 10, 100, 1000, then 5000. The following accuracies were produced:

#### `max_features= 10`:

> The training accuracy is:  0.8715     
> The validation accuracy is:  0.5604

#### `max_features= 100`:

> The training accuracy is:  0.9999       
> The validation accuracy is:  0.7212

#### `max_features= 1000`:

> The training accuracy is:  1.0     
> The validation accuracy is:  0.8216

#### `max_features= 5000`:

> The training accuracy is:  1.0       
> The validation accuracy is:  0.8384

__Brief Overview__:
Increasing the `max_features` parameter seems to increase the accuracy of the model, however that increase slows as n increases.

---

# Overall Summary

Stemming and lemmatizing the words before training the model does not appear to have a significant impact on the accuracy, despite the amount of time it takes to compute these word changes. Likewise, switching to 2-gram bag of words features also doesn't seem to have a significant imacte on the accuracy of the model. Of all the modifications made, the most significant change in accuracy came from altering the `max features` setting. The validation accuracy increased with the number of features, which is to be expected as more features leads to a more robust decision tree framework in the random forest algorithm. However, this increase does not grow proportionally with the number of features and more features means a more biased model. The original 1000 `max_features` was only 2% less accurate than the modified 5000 `max_features` and provided less biased results. In conclusion, the methods used in this experiment did not seem to have a significant impact on the accuracy of the model. Other methods may be more useful and less computationally expensive.
