# career_village_entities
Object model for the
[Kaggle Data Science for Good: CareerVillage.org](https://www.kaggle.com/c/data-science-for-good-careervillage) 
initiative.

## Getting started
First, we want to read in the CSV files to build the
object model. We then save that to a pickle file.
```python
from career_village_entities import CareerVillage

# Load the raw data and save as a pickle file
CareerVillage.load_raw('input/').save('data/cv.p')

# In the future, we can just load the pickle
cv = CareerVillage.load('data/cv.p')

```

After that, we can just load the pickle file.

```python
from career_village_entities import CareerVillage

# In the future, we can just load the pickle
cv = CareerVillage.load('data/cv.p')
```

The CareerVillage instances contains several collections, one
for each type of entity in the data set.
Each collection is a
[scalaps](https://github.com/matthagy/scalaps)
list, which gives it a lot of useful helper methods.
```python
from career_village_entities import CareerVillage

cv = CareerVillage.load('data/cv.p')

print(cv.tags.length, 'tags')

cv.questions.take(5).for_each(print)
```

Each entity is linked to other entities. E.g., an `Answer`
is linked to it's question and it's author.
Similarly, each person (`Student` or `Professional`)
is linked to the questions they've asked and the answers
they've provided.
This helps us find patterns in the data for use
in developing methods to recommend specific questions
to specific professionals.

Here's an example where we check how important emails are
for encouraging answers.
We simply check how frequently a question was answered
by a professional who was emailed with a suggestion to
answer that question.
```python
# Count how many questions were answered by a professional emailed about the question
# vs. how many questions were answered w/o prompting
from career_village_entities import CareerVillage

cv = CareerVillage.load('data/cv.p')

def is_question_answered_by_emailed_professional(question):
    emailed_professionals = question.emails.map('recipient')
    authors = question.answers.map('author')
    return bool(set(emailed_professionals) & set(authors))

(cv
 .questions
 .filter(lambda q: q.answers.length > 0) # Only consider questions that were answered
 .group_by(is_question_answered_by_emailed_professional)
 .items()
 .map(lambda x: (x[0], x[1].length))
 .for_each(print))
```

The results are
```
(True, 10452)
(False, 12658)
```
Hence, 45.2% of answered questions were answered in repsponse to
an email prompt.


Very much a work in progress. 
I'd very match appreciate other people's input, so feel free to submit a PR.

Contact: Matt Hagy <matthew.hagy@gmail.com>