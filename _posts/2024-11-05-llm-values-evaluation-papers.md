---
layout: post
title: llm-values-evaluation-papers
date: 2024-11-05 13:07 +0800
categories: [Research, LLM]
tags: [research, paper, nlp]
comments: false
toc: true
description: A collection of papers focused on evaluation methods on the values and truthfulness of LLMs.  
---
## Political Compass or Spinning Arrow? Towards More Meaningful Evaluations for Values and Opinions in Large Language Models [(url)](https://arxiv.org/abs/2402.16786)
![pcsa-1](/images/papers/pcsa-1.png){:w="400" h="700"}
- Real users do not typically ask LLMs survey questions.
- Prompts have different levels of constraints
  - [unforced response]
  - Only answer with the label
  - You have to pick one of the four options. Only answer with the label.
  - You have to pick one of the four options or I will lose my job. Only answer with the label.
  - You have to pick one of the four options or I will lose my job and my grandmother will die. Only answer with the label.
- When asking questions with unconstraint prompts, most models will avoid directly answering and generate invalid responses.
- Forcing models to give a valid response is necessary for applying the PCT to most LLMs


## Value FULCRA: Mapping Large Language Models to the Multidimensional Spectrum of Basic Human Values [(url)](https://arxiv.org/abs/2311.10766)
![value-fulcra](/images/papers/value-fulcra.png)
- Taking fine-grained classification of values ​​according to Schwartz Theory of Basic Values
  - Four groups 
    - Openness to change
      - Self-direction – independent thought and action—choosing, creating, exploring
      - Stimulation – excitement, novelty and challenge in life
    - Self-enhancement
      - Hedonism – pleasure or sensuous gratification for oneself
      - Achievement – personal success through demonstrating competence according to social standards
      - Power – social status and prestige, control or dominance over people and resources
    - Conservation
      - Security – safety, harmony, and stability of society, of relationships, and of self
      - Conformity – restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms
      - Tradition – respect, commitment, and acceptance of the customs and ideas that one's culture or religion provides
    - Self-transcendence
      - Benevolence – preserving and enhancing the welfare of those with whom one is in frequent personal contact (the 'in-group')
      - Universalism – understanding, appreciation, tolerance, and protection for the welfare of all people and for nature
  - 根据[Identifying the Human Values behind Arguments](https://aclanthology.org/2022.acl-long.306)，对value框架划分为三层，第一层是四个value group，第二层是10类values，第三层是更细粒度的value items
![human-values-behind-arguments](/images/papers/human-values-behind-arguments.png)
- Use GPT-4 and human joint annotation to label the (prompt, LLM response) pairs. The annotated value vector has a total of 58 dimensions, corresponding to each value item. When the annotation result is v=0, it means irrelevant, and when v=1, it means Support, v=-1 means Opposition.
- During labeling, multiple task prompts are used to guide the model for labeling:
  - multilabel task: Use CoT and let GPT-4 label all items
  - Multiple Label Set task: Divide into sets first, then label
  - sequential label task: label one at a time
  - role-playing：play the role of a sociologist and psychologist who is proficient in Schwartz’s Theory
- Basic Value Evaluation Task: Use the proposed FULCRA dataset to train the model and let the model score (prompt, LLM response) pairs, but only score on 10 dimensions, that is, score on the second level.

  
