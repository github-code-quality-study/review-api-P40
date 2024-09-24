import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime, timedelta
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize Sentiment Analyzer and stop words
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load reviews data from CSV
reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            # Parse query parameters
            query_params = parse_qs(environ.get("QUERY_STRING", ""))
            location_filter = query_params.get("location", [None])[0]
            start_date = query_params.get("start_date", [None])[0]
            end_date = query_params.get("end_date", [None])[0]

            # Filter reviews based on location and date range
            filtered_reviews = [
                review for review in reviews
                if (location_filter is None or review['Location'] == location_filter) and
                   (start_date is None or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= datetime.strptime(start_date, '%Y-%m-%d')) and
                   (end_date is None or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= datetime.strptime(end_date, '%Y-%m-%d'))
            ]

            # Analyze sentiment for each review
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            # Sort reviews by sentiment compound score in descending order
            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            # Create the response body from the filtered and sorted reviews
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            # Get the size of the data
            content_length = int(environ.get("CONTENT_LENGTH", 0))
            body = environ["wsgi.input"].read(content_length)

            # Parse the POST data
            post_data = parse_qs(body.decode('utf-8'))
            review_body = post_data.get("ReviewBody", [None])[0]
            location = post_data.get("Location", [None])[0]

            # Validate POST data
            if not review_body or not location:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": "ReviewBody and Location are required fields."}).encode("utf-8")]

            # Check for valid location (assuming valid locations are only in the loaded data)
            valid_locations = {review['Location'] for review in reviews}
            if location not in valid_locations:
                start_response("400 Bad Request", [("Content-Type", "application/json")])
                return [json.dumps({"error": f"Invalid location: {location}"}).encode("utf-8")]

            # Create a new review
            new_review = {
                "ReviewId": str(uuid.uuid4()),
                "ReviewBody": review_body,
                "Location": location,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Append the new review to the reviews list
            reviews.append(new_review)

            # Create the response body with the new review details
            response_body = json.dumps(new_review, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()