

# Wikipedia Search Engine

This project is a search engine built for English Wikipedia as part of the Information Retrieval course (2023-2024). The engine supports multiple ranking methods and is designed for efficient and accurate query processing. It was developed to meet course requirements, including deploying on Google Cloud Platform (GCP).

---

## Features

1. **Search Functionality**
   - Processes user queries to return relevant Wikipedia articles.
   - Implements six ranking methods:
     - Cosine similarity using TF-IDF on the body of articles.
     - Binary ranking using the title of articles.
     - Binary ranking using anchor text.
     - Ranking by PageRank.
     - Ranking by article page views.

2. **Efficiency**
   - Average query retrieval time optimized to meet the requirement of ≤ 35 seconds.

3. **Quality**
   - Ensures an Average Precision@10 > 0.1 on the test set.

4. **Deployment**
   - Fully functional on GCP.
   - Accessible through a public URL for testing and evaluation.

---

## Project Structure

```
├── code/
│   ├── search_frontend.py       # Flask app for search engine frontend.
│   ├── run_frontend_in_colab.ipynb  # Notebook for local development using Colab.
│   ├── run_frontend_in_gcp.sh   # Script for deploying on GCP.
│   ├── startup_script_gcp.sh    # GCP instance setup script.
│   ├── inverted_index_gcp.py    # Inverted index implementation for GCP.
│   └── utils/                   # Helper functions and utilities.
├── data/
│   ├── queries_train.json       # Training queries with ranked results.
│   ├── wikipedia_dump/          # Processed Wikipedia dump.
│   └── pageviews/               # Pageview statistics for articles.
├── docs/
│   └── report.pdf               # Detailed project report.
├── tests/
│   ├── test_queries.py          # Automated tests for query processing.
│   └── test_index.py            # Automated tests for index creation.
└── README.md                    # Project documentation.
```

---

## Setup and Installation

### Requirements

- Python 3.8 or higher
- Flask
- Google Cloud SDK
- Required Python packages: `requirements.txt`

### Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask frontend:
   ```bash
   python code/search_frontend.py
   ```

4. Access the local development server at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Deploying to GCP

1. Start a Compute Engine instance and reserve a public IP using the provided script:
   ```bash
   bash run_frontend_in_gcp.sh
   ```

2. Use the public IP to access the deployed engine.

---

## Usage

1. Access the search engine through the frontend (either locally or via GCP).
2. Enter a query in the search bar to retrieve relevant Wikipedia articles.
3. Results are ranked based on the selected ranking method.

---

## Evaluation

- **Performance Metrics**
  - Precision@10 and F1@30 evaluated on test queries.
  - Graphical analysis of retrieval time and performance for different ranking methods.
  
- **Qualitative Analysis**
  - Case studies of queries with excellent and poor results.

---


## Main Retrieval Methods

The search engine supports multiple retrieval methods, ensuring diverse approaches to ranking and retrieving results:

- **TF-IDF**: Calculates term frequency-inverse document frequency for relevance scoring.
- **BM-25**: A probabilistic retrieval model that enhances traditional tf-idf.
- **Cosine Similarity**: Measures similarity between the query and documents in the vector space.
- **Binary Search**:
  - With stemming: Processes words to their root forms for improved matches.
  - Without stemming: Matches words exactly as they appear.
- **Page Rank**: Utilizes a graph-based ranking algorithm to assess the importance of articles.
- **Page View**: Ranks articles based on their popularity using page view statistics.

---

## Indexes

Efficient retrieval is enabled by multiple indexes, each serving a specific purpose:

- **Title Index**: Indexes article titles (available with and without stemming).
- **Body Index**: Contains the full text of articles.
- **Anchor Text Index**: Captures anchor text from links for context-based retrieval.

---

## Dictionaries

Several dictionaries are utilized to enhance retrieval efficiency:

- **Docs_len_Body_Dict**: Stores document lengths for body text.
- **id_title_dict**: Maps document IDs to their titles.
- **norm_dict**: Contains normalization factors for cosine similarity calculations.
- **page_rank**: Stores pre-computed PageRank scores for articles.
- **pageviews**: Contains page view statistics for articles.

---

## Achievement

This project was awarded **the best project in class** due to the high quality of search results produced by our search engine.

Feel free to explore the code and experiment with different retrieval methods and search queries!

---

## Acknowledgments

Special thanks to the Information Retrieval course team for providing the tools and datasets to build this project.


---

## License

This project is developed as part of the academic curriculum. Please refer to the course guidelines for usage restrictions.

---

