FROM python:3.8
# python:alpine will not work!
COPY requirement.txt .

RUN pip install -r requirement.txt

RUN python -c "import nltk; nltk.download('omw-1.4'); nltk.download('wordnet')"

COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]