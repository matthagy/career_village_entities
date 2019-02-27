"""Object model for the Kaggle Data Science for Good: CareerVillage.org initiative.
See https://www.kaggle.com/c/data-science-for-good-careervillage

Library currently a work in progress and improvements are much appreciated.
Feel free to send a PR on GitHub.
"""

# Copyright 2019 Matt Hagy <matthew.hagy@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the “Software”), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import csv
import pickle
import sys
from datetime import datetime
from typing import Callable, Union, Iterable

from scalaps import ScSeq, ScList, ScFrozenList, ListMixin

NoneType = type(None)


def read_csv(path: str) -> Iterable[dict]:
    with open(path) as fp:
        reader = iter(csv.reader(fp))
        header = next(reader)
        for row in reader:
            yield dict(zip(header, row))


def load_seq(path: str, func: Callable) -> ScFrozenList:
    return ScFrozenList(func(d) for d in read_csv(path))


def empty_string_to_none(s: str) -> Union[str, None]:
    return None if not s else s


def quick_parse_datetime(s) -> datetime:
    """ Assume format like '2011-10-05 20:35:19 UTC+0000'
    """
    d, t, s = s.split(' ')
    assert s == 'UTC+0000'
    year, month, day = map(int, d.split('-'))
    hour, minute, second = map(int, t.split(':'))
    return datetime(year, month, day, hour, minute, second)


class BaseEntity:
    def __repr__(self):
        scalars, lists = self.partitioned_attrs()
        components = []
        for k, v in sorted(scalars.items()):
            components.append(f'{k}={v!r}')
        for k, v in sorted(lists.items()):
            components.append(f'{k}={v.length}')
        return f'<{self.__class__.__name__} {" ".join(components)}>'

    def attrs(self):
        return {k: v for k, v in vars(self).items()
                if not k.startswith('_') and not k.endswith('_id')}

    def partitioned_attrs(self):
        scalars = {}
        lists = {}
        for k, v in self.attrs().items():
            (lists if isinstance(v, ListMixin) else scalars)[k] = v
        return scalars, lists

    def to_json(self):
        scalars, lists = self.partitioned_attrs()
        components = {}
        for k, v in sorted(scalars.items()):
            components[k] = v.to_shallow_json() if hasattr(v, 'to_shallow_json') else v
        for k, v in sorted(lists.items()):
            components[k] = v.length
        return {self.__class__.__name__: components}

    def to_shallow_json(self):
        scalars, lists = self.partitioned_attrs()
        components = {}
        for k, v in sorted(scalars.items()):
            if isinstance(v, (str, int, datetime, NoneType)):
                components[k] = v
        for k, v in sorted(lists.items()):
            components[k] = v.length
        return {self.__class__.__name__: components}


class BaseHasUsers(BaseEntity):
    def __init__(self):
        self.users = ScList()

    def _freeze(self):
        self.users = self.users.to_frozen_list()
        return self

    @property
    def students(self):
        return ScSeq(u for u in self.users if isinstance(u, Student))

    @property
    def professionals(self):
        return ScSeq(u for u in self.users if isinstance(u, Professional))


class Tag(BaseHasUsers):
    def __init__(self, tags_id: int, name: str):
        super().__init__()
        self.tags_id = tags_id
        self.name = name
        self.questions = ScList()

    def _freeze(self):
        super()._freeze()
        self.questions = self.questions.to_frozen_list()
        return self

    @classmethod
    def load(cls, path) -> ScFrozenList:
        return load_seq(path, lambda d: Tag(int(d['tags_tag_id']), d['tags_tag_name']))


class Group(BaseHasUsers):
    def __init__(self, groups_id: str, group_type: str):
        super().__init__()
        self.groups_id = groups_id
        self.group_type = group_type

    @classmethod
    def load(cls, path) -> ScFrozenList:
        return load_seq(path, lambda d: Group(d['groups_id'], d['groups_group_type']))


class School(BaseHasUsers):
    def __init__(self, school_id: int):
        super().__init__()
        self.school_id = school_id


class BasePerson(BaseEntity):
    def __init__(self, user_id: str, location: Union[str, NoneType], date_joined: datetime):
        self.user_id = user_id
        self.location = location
        self.data_joined = date_joined

        self.tags = ScList()
        self.groups = ScList()
        self.schools = ScList()
        self.questions = ScList()
        self.answers = ScList()

    def _freeze(self):
        self.tags = self.tags.to_frozen_list()
        self.groups = self.groups.to_frozen_list()
        self.schools = self.schools.to_frozen_list()
        self.questions = self.questions.to_frozen_list()
        self.answers = self.answers.to_frozen_list()
        return self


class Student(BasePerson):
    def __init__(self, students_id: str, location: Union[str, NoneType], date_joined: datetime):
        super().__init__(students_id, location, date_joined)

    @property
    def students_id(self):
        return self.user_id

    @classmethod
    def load(cls, path):
        return load_seq(path, lambda d: Student(
            d['students_id'],
            empty_string_to_none(d['students_location']),
            quick_parse_datetime(d['students_date_joined'])))


class Professional(BasePerson):
    def __init__(self,
                 professionals_id: str,
                 location: Union[str, NoneType],
                 industry: Union[str, NoneType],
                 headline: Union[str, NoneType],
                 date_joined: datetime):
        super().__init__(professionals_id, location, date_joined)
        self.professionals_id = professionals_id
        self.industry = industry
        self.headline = headline

        self.emails = ScList()

    def _freeze(self):
        super()._freeze()
        self.emails = self.emails.to_frozen_list()
        return self

    @classmethod
    def load(cls, path):
        return load_seq(path, lambda d: Professional(
            d['professionals_id'],
            empty_string_to_none(d['professionals_location']),
            empty_string_to_none(d['professionals_industry']),
            empty_string_to_none(d['professionals_headline']),
            quick_parse_datetime(d['professionals_date_joined'])))


class Question(BaseEntity):
    author = None  # Changed in instances when linked

    def __init__(self,
                 questions_id: str,
                 author_id: str,
                 date_added: datetime,
                 title: str,
                 body: str):
        self.questions_id = questions_id
        self.author_id = author_id
        self.date_added = date_added
        self.title = title
        self.body = body

        self.tags = ScList()
        self.emails = ScList()
        self.answers = ScList()

    def _freeze(self):
        self.tags = self.tags.to_frozen_list()
        self.emails = self.emails.to_frozen_list()
        self.answers = self.answers.to_frozen_list()
        return self

    @classmethod
    def load(cls, path):
        return load_seq(path, lambda d: Question(
            d['questions_id'],
            d['questions_author_id'],
            quick_parse_datetime(empty_string_to_none(d['questions_date_added'])),
            d['questions_title'],
            d['questions_body']))


class Answer(BaseEntity):
    # Changed in instances when linked
    author = None
    question = None

    def __init__(self,
                 answers_id: str,
                 author_id: str,
                 question_id: str,
                 date_added: datetime,
                 body: str):
        self.answers_id = answers_id
        self.author_id = author_id
        self.question_id = question_id
        self.date_added = date_added
        self.body = body

    def _freeze(self):
        return self

    @classmethod
    def load(cls, path):
        return load_seq(path, lambda d: Answer(
            d['answers_id'],
            d['answers_author_id'],
            d['answers_question_id'],
            quick_parse_datetime(empty_string_to_none(d['answers_date_added'])),
            d['answers_body']))


class Email(BaseEntity):
    recipient = None

    def __init__(self,
                 emails_id: str,
                 recipient_id: str,
                 date_sent: datetime,
                 frequency_level: str):
        self.emails_id = emails_id
        self.recipient_id = recipient_id
        self.date_sent = date_sent
        self.frequency_level = frequency_level

        self.questions = ScList()

    def _freeze(self):
        self.questions = self.questions.to_frozen_list()
        return self

    @classmethod
    def load(cls, path):
        return load_seq(path, lambda d: Email(
            d['emails_id'],
            d['emails_recipient_id'],
            quick_parse_datetime(empty_string_to_none(d['emails_date_sent'])),
            d['emails_frequency_level']))


def freeze_list(l):
    return ScFrozenList(x._freeze() for x in l)


def json_list(l):
    return ScFrozenList(x.to_json() for x in l)


class CareerVillage(BaseEntity):
    def __init__(self, directory_path, tags, groups,
                 students, professionals,
                 questions, answers,
                 emails):
        self.directory_path = directory_path
        self.tags = tags
        self.groups = groups
        self.students = students
        self.professionals = professionals
        self.questions = questions
        self.answers = answers
        self.emails = emails

        self.linked = False

    @classmethod
    def load_raw(cls, directory_path='data') -> 'CareerVillage':
        """Load the data that goes into a CareerVillage.

        Note, you still need to call CareerVillage.link() after load to
        associate together entities. Linking is deferred because the linked
        entities are too heavily nested to be pickeled. Hence, we can
        load the data once, pickle it, and then reload it with pickle.
        """
        tags = Tag.load(directory_path + '/tags.csv')
        print(tags.length, 'tags')
        tags.take(5).for_each(print)

        groups = Group.load(directory_path + '/groups.csv')
        print(groups.length, 'groups')
        groups.take(5).for_each(print)

        students = Student.load(directory_path + '/students.csv')
        print(students.length, 'students')
        students.take(5).for_each(print)

        professionals = Professional.load(directory_path + "/professionals.csv")
        print(professionals.length, 'professionals')
        professionals.take(5).for_each(print)

        questions = Question.load(directory_path + "/questions.csv")
        print(questions.length, 'questions')
        questions.take(5).for_each(print)

        answers = Answer.load(directory_path + "/answers.csv")
        print(answers.length, 'answers')
        answers.take(5).for_each(print)

        emails = Email.load(directory_path + '/emails.csv')
        print(emails.length, 'emails')
        emails.take(5).for_each(print)

        return cls(directory_path,
                   tags, groups,
                   students, professionals,
                   questions, answers,
                   emails)

    def save(self, path='data/cv.p'):
        if self.linked:
            raise RuntimeError("Can't pickle a linked CareerVillage")
        with open(path, 'wb') as fp:
            pickle.dump(self, fp, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path='data/cv.p', link=True) -> 'CareerVillage':
        with open(path, 'rb') as fp:
            cv = pickle.load(fp)
        if link:
            cv.link()
        return cv

    def link(self):
        if self.linked:
            return
        self.linked = True

        students_by_id = self.students.key_by('students_id')
        professionals_by_id = self.professionals.key_by('professionals_id')
        users_by_id = students_by_id.union(professionals_by_id, error_on_overlap=True)

        print('linking tags with users')
        tags_by_id = self.tags.key_by('tags_id')
        for tag_user in read_csv(self.directory_path + '/tag_users.csv'):
            tag = tags_by_id[int(tag_user['tag_users_tag_id'])]
            user = users_by_id[tag_user['tag_users_user_id']]
            tag.users.append(user)
            user.tags.append(tag)

        print('linking groups with users')
        groups_by_id = self.groups.key_by('groups_id')
        for group_membership in read_csv(self.directory_path + '/group_memberships.csv'):
            group = groups_by_id[group_membership['group_memberships_group_id']]
            user = users_by_id[group_membership['group_memberships_user_id']]
            group.users.append(user)
            user.groups.append(group)

        print('loading and linking schools with users')
        schools_by_id = {}
        for school_membership in read_csv(self.directory_path + '/school_memberships.csv'):
            school_id = int(school_membership['school_memberships_school_id'])
            try:
                school = schools_by_id[school_id]
            except KeyError:
                school = schools_by_id[school_id] = School(school_id)

            user = users_by_id[school_membership['school_memberships_user_id']]
            school.users.append(user)
            user.schools.append(school)

        print('linking tags with questions')
        questions_by_id = self.questions.key_by('questions_id')
        for tag_question in read_csv(self.directory_path + '/tag_questions.csv'):
            tag = tags_by_id[int(tag_question['tag_questions_tag_id'])]
            question = questions_by_id[tag_question['tag_questions_question_id']]
            question.tags.append(tag)
            tag.questions.append(question)

        print('linking questions w/ author')
        for question in self.questions:
            # For some reason there are questions that don't have an author
            question.author = users_by_id.get(question.author_id, None)
            if question.author:
                question.author.questions.append(question)

        print('linking answers to authors and questions')
        for answer in self.answers:
            # For some reason there are answers that don't have an author
            answer.author = users_by_id.get(answer.author_id, None)
            if answer.author is not None:
                answer.author.answers.append(answer)

            answer.question = questions_by_id[answer.question_id]
            answer.question.answers.append(answer)

        print('linking emails and recipients')
        for email in self.emails:
            professional = professionals_by_id[email.recipient_id]
            professional.emails.append(email)
            email.recipient = professional

        print('linking questions and emails')
        emails_by_id = self.emails.key_by('emails_id')
        for match in read_csv(self.directory_path + '/matches.csv'):
            email = emails_by_id[match['matches_email_id']]
            question = questions_by_id[match['matches_question_id']]
            email.questions.append(question)
            question.emails.append(email)

        print('freezing')
        self.tags = freeze_list(self.tags)
        self.groups = freeze_list(self.groups)
        self.schools = freeze_list(schools_by_id.values())
        self.students = freeze_list(self.students)
        self.professionals = freeze_list(self.professionals)
        self.questions = freeze_list(self.questions)
        self.answers = freeze_list(self.answers)
        self.emails = freeze_list(self.emails)


def main():
    if __name__ == '__main__':
        print('running with appropriate name space')
        from career_village_entities import main
        return main()

    sys.setrecursionlimit(1000000000)
    print('loading raw')
    cv = CareerVillage.load_raw()
    print('saving pickle')

    print('loading pickle for demo')
    cv = CareerVillage.load()

    from pprint import pprint
    json_list(cv.tags.take(5)).for_each(pprint)
    json_list(cv.groups.take(5)).for_each(pprint)
    json_list(cv.schools.take(5)).for_each(pprint)
    json_list(cv.students.take(5)).for_each(pprint)
    json_list(cv.professionals.take(5)).for_each(pprint)
    json_list(cv.questions.take(5)).for_each(pprint)
    json_list(cv.answers.take(5)).for_each(pprint)
    json_list(cv.emails.take(5)).for_each(pprint)


__name__ == '__main__' and main()
