import joblib
import pandas as pd

class TicketPriorityPredictor:
    """Predict priority for new tickets."""
    def __init__(self):
        self.model = joblib.load("models/priority_model.pkl")
        self.tfidf = joblib.load("models/tfidf_vectorizer.pkl")
        self.priority_map = joblib.load("models/priority_map.pkl")
    
    def predict(self, summary, description):
        """Predict priority for a single ticket."""
        text = summary + " " + description
        vector = self.tfidf.transform([text])
        priority_code = self.model.predict(vector)[0]
        return self.priority_map[priority_code]
    
    def batch_predict(self, df):
        """Predict priorities for a DataFrame of tickets."""
        df["text"] = df["summary"] + " " + df["description"]
        vectors = self.tfidf.transform(df["text"])
        df["predicted_priority"] = [self.priority_map[code] for code in self.model.predict(vectors)]
        return df