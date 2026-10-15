"""Datasets (1 = positive, 0 = negative).

TEST_*  : hand-written sentences, never used for training (honest evaluation).
TRAIN_* : generated from templates with varied subjects/opinions. Replace with real
          labelled data (e.g. product reviews) for a serious model.
"""
import random
TEST_POSITIVE = [
    "I absolutely love this product", "This is fantastic, works perfectly", "Great quality and fast delivery",
    "Really happy with my purchase", "Excellent service, highly recommend", "The food was delicious and fresh",
    "What a wonderful experience", "Amazing value for the price", "The staff were friendly and helpful",
    "I am very satisfied with the results", "Best purchase I have made this year", "Brilliant, exactly what I needed",
    "The movie was thrilling and beautifully made", "Super easy to use and reliable", "I enjoyed every minute of it",
    "Outstanding performance and great battery life", "Lovely place, we will come back", "Top notch quality, very impressed",
    "The app is smooth and intuitive", "Five stars, couldn't be happier", "Delightful and well designed",
    "It exceeded my expectations", "Very comfortable and stylish", "The support team solved my issue quickly",
    "A perfect gift, she loved it", "Clean, modern and spotless room", "Great teacher, explains things clearly",
    "This update made everything better", "Terrific sound quality", "I'm thrilled with how it turned out",
]
TEST_NEGATIVE = [
    "I hate this product", "This is terrible, it broke immediately", "Awful quality and very slow delivery",
    "Really disappointed with my purchase", "Worst service I have ever had", "The food was cold and tasteless",
    "What a horrible experience", "Total waste of money", "The staff were rude and unhelpful",
    "I am very unhappy with the results", "Worst purchase I have made this year", "Useless, not what I needed at all",
    "The movie was boring and badly acted", "Confusing to use and unreliable", "I regret buying it",
    "Poor performance and terrible battery life", "Dirty place, we will never return", "Cheap quality, very disappointing",
    "The app is buggy and keeps crashing", "One star, completely unsatisfied", "Ugly and poorly designed",
    "It fell far short of my expectations", "Uncomfortable and overpriced", "The support team ignored my issue",
    "A bad gift, she returned it", "Filthy and noisy room", "Terrible teacher, explains nothing",
    "This update made everything worse", "Awful sound quality", "I'm annoyed with how it turned out",
]

TEST_TEXTS = TEST_POSITIVE + TEST_NEGATIVE
TEST_LABELS = [1] * len(TEST_POSITIVE) + [0] * len(TEST_NEGATIVE)

_SUBJECTS = ["the hotel", "this phone", "the pizza", "our teacher", "the delivery", "this laptop", "the film",
             "the new update", "the customer service", "this book", "the restaurant", "my headphones", "the course",
             "this jacket", "the game", "the website", "the concert", "this blender", "the repair shop", "the cafe",
             "the app", "this car", "the room", "the staff", "my order", "the show", "this product", "the service",
             "the food", "the camera", "the seats", "this sofa", "the flight", "the tutorial", "the manager"]
_POS = ["is excellent", "was wonderful", "is absolutely fantastic", "was a pleasure", "is superb",
        "works great and I love it", "was amazing, highly recommended", "is really good value", "was impressive",
        "is brilliant", "made me very happy", "was better than expected", "is outstanding", "feels premium and reliable",
        "was delightful from start to finish", "is a joy to use", "was great, five stars", "is perfect for me",
        "is lovely and I would buy it again", "was fast, friendly and helpful", "is beautiful and well made",
        "left me thrilled and satisfied", "is worth every penny", "was fun and enjoyable", "is smooth, easy and reliable",
        "was so good that I recommend it to everyone", "exceeded my expectations", "is clean, comfortable and stylish",
        "was a great surprise, I am impressed", "is the best I have tried", "was top quality and I enjoyed it",
        "is fabulous and I'm very pleased", "gave me a great experience", "is nice, pleasant and satisfying"]
_NEG = ["is terrible", "was awful", "is absolutely dreadful", "was a nightmare", "is poor",
        "stopped working and I hate it", "was horrible, avoid it", "is a rip-off", "was disappointing",
        "is rubbish", "made me very angry", "was worse than expected", "is mediocre at best", "feels cheap and unreliable",
        "was frustrating from start to finish", "is a pain to use", "was bad, one star", "is useless to me",
        "is ugly and I would not buy it again", "was slow, rude and unhelpful", "is flimsy and badly made",
        "left me upset and unsatisfied", "is a waste of money", "was boring and tedious", "is buggy, confusing and unreliable",
        "was so bad that I do not recommend it to anyone", "fell short of my expectations", "is dirty, noisy and uncomfortable",
        "was a nasty surprise, I am disappointed", "is the worst I have tried", "was poor quality and I regret it",
        "is dreadful and I'm very annoyed", "gave me a miserable experience", "is unpleasant, annoying and unsatisfying"]
_FRAMES = ["{s} {o}", "Honestly, {s} {o}", "I think {s} {o}", "Overall {s} {o}", "{s} {o}, no doubt", "To be fair, {s} {o}",
           "Wow, {s} {o}", "In my opinion {s} {o}", "{s} {o} and that is the truth", "Well, {s} {o}", "Sadly or not, {s} {o}"]

def _make(opinions, label, n, rng):
    rows = set()
    while len(rows) < n:
        s, o, f = rng.choice(_SUBJECTS), rng.choice(opinions), rng.choice(_FRAMES)
        t = f.format(s=s, o=o)
        rows.add(t[0].upper() + t[1:])
    return list(rows), [label] * n

_rng = random.Random(7)
_pt, _pl = _make(_POS, 1, 400, _rng)
_nt, _nl = _make(_NEG, 0, 400, _rng)
TRAIN_TEXTS = _pt + _nt
TRAIN_LABELS = _pl + _nl
