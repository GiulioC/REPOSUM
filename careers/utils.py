from sklearn.linear_model import LogisticRegression

ucols = {
    "Dummett": {
        "uni-src": 6,
        "rank-src": 7,
        "uni-dst": 12,
        "rank-dst": 13,
        "success": 14
    },
    "Wittgenstein": {
        "uni-src": 4,
        "rank-src": 5,
        "uni-dst": 13,
        "rank-dst": 14,
        "success": 18
    },
    "Lewis": {
        "uni-src": 5,
        "rank-src": 6,
        "uni-dst": 11,
        "rank-dst": 12,
        "success": 15
    },
    "Gadamer": {
        "uni-src": 6,
        "rank-src": 7,
        "uni-dst": 12,
        "rank-dst": 13,
        "success": 14
    },
    "Kripke": {
        "uni-src": 6,
        "rank-src": 7,
        "uni-dst": 12,
        "rank-dst": 13,
        "success": 15
    },
    "Fodor": {
        "uni-src": 6,
        "rank-src": 7,
        "uni-dst": 12,
        "rank-dst": 13,
        "success": 14
    },
    "Spinoza": {
        "uni-src": 5,
        "rank-src": 6,
        "uni-dst": 12,
        "rank-dst": 13,
        "success": 14
    }
}

def get_columns_info(philosopher):
    return ucols[philosopher]

def load_estimator():
    return LogisticRegression(
        solver = 'lbfgs',
        multi_class = 'multinomial',
        max_iter = 1000,
        n_jobs = -1
    )
