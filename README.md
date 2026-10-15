# Sentiment Classifier (starter AI model)

TF-IDF features + Logistic Regression that labels English sentences as positive or negative.

## Setup
    pip install -r requirements.txt

## Train
    python train.py

## Predict
    python predict.py "The service was wonderful"
    python predict.py            # interactive mode

## Notes
- The built-in dataset is synthetic (template-generated), so treat it as a demo. Add many more
  real examples to data.py, or load a real dataset (e.g. product reviews), for real use.
- To go further, swap the pipeline for a fine-tuned transformer.
