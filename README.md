**Using NLP to Predict Song Genres through their Lyrics**

**Introduction**

Natural Language Processing (NLP) techniques allow making sense of natural language, one of the most challenging types of data in machine learning given the unbounded, unstructured nature of language. In an era of digital circulation and streaming, music metadata is crucial in the process of accurately tagging and classifying music to aid its discoverability. How might we leverage NLP techniques in order to correctly assign genre labels to songs? This project combines NLP techniques with neural networks in order to predict genre labels on a corpus of song lyrics.

A [2019 report](https://www.musicbusinessworldwide.com/nearly-40000-tracks-are-now-being-added-to-spotify-every-single-day/) published by Music Business Worldwide asserts that nearly 40,000 tracks are uploaded to Spotify every day. Given such a volume, manual classification is an expensive and slow task that is also error-prone. While in theory those uploading bear the burden of providing meaningful and accurate metadata, automating a process for sanity checks is well worth the investment. This notebook explores some ways to implement a genre-tagging model that is based on neural network training of labeled lyrics datasets.

Stakeholders: Who might be interested in this report?

A wide range of audiences are invested in understanding how to automate genre label tagging, including:

- Artists who upload music to various online platforms and hope to maximize their searchability and findability
- The platforms themselves, who hope to optimize music discovery and recommendation processes
- Audiences who want to find music according to genre labels and also discover new acts that might be of interest
