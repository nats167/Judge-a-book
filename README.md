# Judge-a-book

Judge a Book is a machine learning based book recommendation app that suggests books based on user preferences and uploaded book cover images. It uses OCR to extract the book title from the image and then applies a content based filtering approach, leveraging features like author and genre to find and recommend similar books. The app combines basic image processing and machine learning techniques to create a simple and interactive recommendation system.


## Machine learning and python

The recommendation engine is implemented using Python, using libraries such as pandas, scikit learn, and pytesseract. Python handles all data preprocessing tasks like cleaning the books dataset, building combined feature strings from title, author, genre, and description, and computing a TF-IDF matrix used as input to the cosine similarity model. The similarity matrix is exported as a .pkl file so it can be reloaded without recomputing on every run.


<img width="471" height="168" alt="Screenshot from 2026-05-04 22-21-52" src="https://github.com/user-attachments/assets/ae970380-df98-458f-bdb0-239770e0fdad" />

## Features

- **Book Cover OCR** — extracts title and author from any uploaded image
- **Genre Filter** — Fiction, Mystery, Sci-Fi, Fantasy, Thriller, Romance, Biography, Self-Help, History, Horror
- **AI Recommendations** — 2 curated books with similarity reasons and tags
- **Interactive UI** — clean, responsive web interface
- **Manual Input** — type a book title directly if no image is available

## Technologies Used

The Smart Bookshelf Recommender is built entirely in Python, combining data processing, machine learning, OCR, and an interactive UI layer. All development and testing is done inside Google Colab with no local setup required.



<table border="0">
  <tr>
    <!-- Card 1: Python -->
    <td width="33%" valign="top">
      <div align="left">
        <img src="https://img.icons8.com/color/48/python--v1.png" width="40"/>
        <strong>Python</strong><br>
        <p align="left"><sub>core language</sub></p>
        <p align="left">Primary language for all data processing, ML logic, and OCR pipeline.</p>
      </div>
    </td>
    <!-- Card 2: pandas -->
    <td width="33%" valign="top">
      <div align="left">
        <img src="https://img.icons8.com/color/48/pandas.png" width="40"/>
        <strong>pandas</strong><br>
        <p align="left"><sub>data handling</sub></p>
        <p align="left">Loads and cleans the books dataset. Handles missing values and column renaming.</p>
      </div>
    </td>
    <!-- Card 3: scikit-learn -->
    <td width="33%" valign="top">
      <div align="left">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg"width="40"/>
        <strong>scikit-learn</strong><br>
        <p align="left"><sub>ml model</sub></p>
        <p align="left">Provides TfidfVectorizer and cosine_similarity for the recommendation engine.</p>
      </div>
    </td>
  </tr>
      <!-- Card 4: Pytesseract -->
    <td width="33%" valign="top">
      <div align="left">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSEPu-kPAnqaYfjGSTyHRR_ofy-FzyQX239rw&s" width="40"/>
        <strong>Pytesseract</strong><br>
        <p align="left"><sub>ocr engine</sub></p>
        <p align="left">Extracts text from uploaded book cover images to detect the book title.</p>
      </div>
    </td>
    <!-- Card 5: google colab -->
    <td width="33%" valign="top">
      <div align="left">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbP3Rmtg8v5dpAmpPDOQ1XhbkCgO7QKLGY_A&s" width="40"/>
        <strong>Google colab</strong><br>
        <p align="left"><sub>environment</sub></p>
        <p align="left">Cloud-based notebook environment — no local setup needed, runs in the browse</p>
      </div>
    </td>
    <!-- Card 5: Pillow(PIL) -->
    <td width="33%" valign="top">
      <div align="left">
        <img src="" width="40"/>
        <strong>Pillow(PIL)</strong><br>
        <p align="left"><sub>image processing</sub></p>
        <p align="left">Opens and preprocesses book cover images before feeding them into OCR.</p>
      </div>
    </td>
  </tr>
      <!-- Card 6: Difflib -->
    <td width="33%" valign="top">
      <div align="left">
        <img src="https://img.icons8.com/color/48/python--v1.png" width="40"/>
        <strong>Python</strong><br>
        <p align="left"><sub>fuzzy matching</sub></p>
        <p align="left">Matches the OCR-extracted title to the closest book title in the dataset.</p>
      </div>
    </td>
    <!-- Card 7: Pickle -->
    <td width="33%" valign="top">
      <div align="left">
        <img src="https://img.icons8.com/color/48/pandas.png" width="40"/>
        <strong>Difflib</strong><br>
        <p align="left"><sub>model export</sub></p>
        <p align="left">Saves the similarity matrix as a .pkl file so it doesn't recompute every run.</p>
      </div>
    </td>
   
</table>

## How it works

- **user input**:The user picks a preferred genre (e.g. Fiction, Mystery) and uploads a photo of any book cover. The genre is used later to filter results. The image is passed straight into the OCR step.
- **Image is preprocessed by pillow**:PIL opens the image and converts it to grayscale (.convert('L')). This removes colour noise and makes the text on the cover cleaner and easier for the OCR engine to read accurately.
- **Pytesseract extracts text from the image**:pytesseract.image_to_string() scans the preprocessed image and pulls out all readable text — including the book title, author name, and any other text on the cover. The top lines are used as the title candidate.
- **Extracted text is matched to the dataset**:difflib.get_close_matches() compares the OCR output against every title in the books dataset and finds the closest match — even if the OCR made small spelling mistakes. This gives us the exact book record to work with.
- **TF-IDF and cosine similarity find similar books**:Each book in the dataset is converted into a numeric vector using TfidfVectorizer — based on its title, author, category, and description. cosine_similarity then measures how close the detected book's vector is to every other book, producing a similarity score for each one.
- **Top results are filtered by genre and returned**



