# Course Outline

<!-- Artificial Intelligence in Medical Epidemiology (Aime) -->
<!-- san fransisco bay, boston, toronto, seattle, montreal -> hotbeds of AI -->

**Prerequisite** <br>
Absolutely None

**What you will learn in this course** <br>
In this course, you will learn what AI is and understand its applications and use cases and how it is transforming our lives. You will be exposed to concerns surrounding AI, including ethics, bias, jobs and the impacts on society
    
**Course Link** <br>
This course is taken from EDX from here &rarr; https://courses.edx.org/courses/course-v1:IBM+AI0101EN+1T2020/course/

**NOTE :** This course is made with jupyter notebook. If you want the notebook drop a message at tbhaxor@gmail.com

# What is AI

AI (also known as Augemented Intelligence or Artificial Intelligence) is something that makes machine act more like humans. The main purpose of the birth of AI is to accomplish the tasks that neither humans not machines could do individually.

### Well what is intelligence
From humans perspective, intelligence is something that governs every activity in our body. 

Since this intelligence can only be possessed by the humans, therefore you could say artificial intelligence is a simulated intelligence of humans to make machine immitate like them

### How do AI learn
AI will only learn that thing which you will make them to learn.

Machines are provided an ability for form the learning models that tends to learn from past examples (data, fact or etc)

The learning methods based on type of data/environment we provide are
+ Supervised Learning
+ Unsupervised Learning
+ Reinforcement Learning

### Types of AI Based on **STRENGTH** and **APPLICATION**

**Weak** or **Narrow AI** is that type AI which is applied to specific domain. For example, virtual assistance, language translators, gaming bots, self-driving cars, recommendation systems and more. It cannot learn new things unless you make them learn. After being trained on the set of data, it can only perform specific task on that type of data only.

**Strong** or **Generalized AI** can interact and operate a wide variety of independent and unrelated tasks. Like humans, it can learn new task of their own to solve new problems. It is combination of different AI strategies that learn from experience and can perform at a human level of intelligence.

**Super** or **Consious AI** is an AI with human level consiousness, which would require it to be self-aware. This is because we are not yet able to adequately define what consiousness is, it is unlikely that we will be able to create a consious ai in the near future.


## How AI is related to Academic Subjects

AI is the fusion of many fields of study

+ Computer science and electrical engineering is used to determine how AI can be implemented in software and hardware.
+ Mathematics and statistics determines viable models and learning performance

+ Neuroscience helps in determining how does the human brain learn new things.

+ Psychology and linguistics play a significant role in understanding how an ai might work

+ Philosophy provides guidance on intelligence and ethical considerations


## Application of AI
1. IBM Watson
2. Netflix's recommending system
3. Virtual Assistants
4. Self driving car
5. Text translation
6. Chatbots
7. Speech Synthesis
8. Computer Vision
9. Facial Recoganistion
10. Image Processing
11. Spam Detection
12. Detecting Fradulent Transactions
13. Mechanical Automation by Robotics
14. Text Composition in Email Body by Google
**Impact of AI** <br>
According to a study by PWC, about 16 trillion of GDP will be added between now and 2030 on the basis of AI. It is not just in IT industry, it impacts virtually every industry and aspects of our lives.

# Concepts and Terminology
AI is a cognitive computing which is different from traditional computing. As in traditional computing, a developer has to hard code the algorithm or series of pattern to accomplish the task. This couldn't adapt to the format, type and amount of input data. Thus these types of application becomes obselete as soon as they are being developed.

In case of cognitive computing, the appliction which is AI driven has an ability to adapt to the changing data. In this, the developer of the learning model has to inject the amount of data for the first time to initially design the learning model. These types of application can adapt the upcoming data and may learn new patterns and make further decisions accordingly.

**How we understand something**
1. **Observe** the physical phenomenon and bodies of evidence
2. We draw on what we know to interpret what we have seen to **genrate hypotheses**
3. **Evaluate** which hypotheses are right or wrong
4. **Actinng** accordingly

Wondering how does cognitive computing learns, they do this by interacting with human. Hence it is more on working with linguistics than mathematics / statistics

## Terminology

1. Artificial Intelligence &rarr; It is a branch of computer science dealing with the simulation of intelligent behaviour. The following are the behaviors often demonstrated <br>
    <img src="https://i.ibb.co/rGxVM3h/image.png" width="900">
    
2. Machine Learning &rarr; It is subset of AI that uses computer algorithms to analyze data and make intelligent decisions based on what is has learned, without being explicitly programmed. <br>
    <img src="https://i.ibb.co/hHmwSNT/image.png" width="900">
    
3. Deep Learning &rarr; It is a specialized subset of machine learning that uses layered neural-networks to simulate human level decision-making
    <img src="https://i.ibb.co/qBJ4t4j/image.png" width="900">
    
4. Neural Networks &rarr; Take inspiration from biological neural network. Immitates the human brain
    <img src="https://i.ibb.co/MZNgFcJ/image.png" width="900">


On a broader look, Deep learning is the part of machine learning and machine learning is itself a part of AI

<p style="text-align: center">
    <img src="https://i.ibb.co/d2DPK9v/image.png" />
</p>

## Machine Learning

As discussed earlier, unlike traditional computing, it tends to learn from the data and use that learning to build a prediction model. 

Let's take an example, <br>
Based on given data determine will the heart will fail beating or not.

Now the problem is straightforward asking for either this or that. These type of problems are called **classification**, where the discrete set of results are given.

In this case, the data given might contain `BPM, BM1, AGE, SEX, RESULT` in which **`RESULT`** is the label which will be used for training our data and to determine whether our model is performing well or not. Others are the data based on which decision will be made. 

### Types of Machine Learning

**Supervised Learning** <br>
An algorithm is based on human labeled data. The more sample you provide a supervised learning algorithm, the more precise it becomes in classifying new data. The following are the sub parts of supervised machine learning
+ Regression
+ Classification
+ Neural Networks


**Unsupervised Learning** <br>
Relies of giving the algorithm unlabeled data and letting it to find patterns by itself.

**Reinforcement Learning** <br>
Relies on providing a machine learning algorithm with a set of rules and constraints and letting it learn how to achieve its goals. You define the state, the desired goals and possibles actions to reach its goals. The agent (in this case) is then either rewarded or punished based on the decision it made. In the end, the agent should reach the end of the goal by collecting many rewards.

### Training with Dataset
In machine learning, we take the entire dataset and divide it into 3 subparts
+ Training data &rarr; used for training the model
+ Testing data &rarr; used for evaluating the performance of model after being trained. While training this data should not be injested into the model
+ Validation data &rarr; used for validating our model prediction during the training and ajusting parameters accordingly

## Deep Learning
Deep learning is the layered stucture of algorithms that immitates the functionality of human brain enabling AI systems to continuously learn on the job and improve the quality and accuracy of results.

Unlike the broader term, machine learning, the more you feed in data in the deep learning model, the more will be its accuracy.

When creating a deep learning model, developers and engineers configure the number of layers and the types of  functions that connect the output of each layer to the input of the next. The neural network then runs on the data we provide it throught the layer and adjust the weights and biases to predict the correct label for the data.

## Artificial Neural Networks

Artificial neural networks borrow some ideas from the biological brain in order to immitate its processing. Neural networks take input and learn through a process known as **backpropagation**

Backpropagation uses a set of training data that match known inputs to desired outputs. Firstly, the inputs are plugged into the network, and outputs are determined. Then, error function determines how far the given outputs are determined. Finally, adjustments are made in order to reduce the error.

### Perceptrons
<p style="text-align: center">
    <img src="https://i.ibb.co/m6bc5fS/image.png">
</p>

Perceptrons are the oldest type of neural networks, where the input layers are directly connected to the output layer. They are also called single-layered perceptrons

Suppose in layer there are two nodes with weights <i>w<sub>0</sub></i> and <i>w<sub>1</sub></i> and input values <i>i<sub>0</sub></i> and <i>i<sub>1</sub></i>. The output of the layer will be <code><i>w<sub>0</sub></i>*<i>v<sub>0</sub></i> + <i>w<sub>1</sub></i>*<i>v<sub>1</sub></i></code>

Usually there is a some unique value of the input node, called **bias** which is added when all the values are multiplied. 

Now this will make a raw neural network, where the opinion (value) of each node will be considered while concluding it at output layer. To limit this, the **activation function** is then applied on the node after its value is being calculated.

## Convolutional Neural Networks (CNN)
CNNs are the neural networks that take inspiration from the animal visual cortex. They are good at **detecting simple feature in an image and putting them together to form a complex feature**.

They are useful in applications such as image processing, video recoganition, face tagging, natural language processing.

## Recurrent Neural Network (RNN)
RNNs are called so because they perform same task for every element of the sequence, with a prior outputs feeding subsequent stage inputs

They are useful in applications such as text summerization, predicting movie scene based on previous context, and anywhere you want the model to remember the context of the input for some time.

When it comes too machine learning, natural langauge processing is the very complex task. As we have make this in such a way a human can only understands it perfectly. As humans we don't view natural language literally, we view it conceptually. For example, for use The and Th3 looks similar, but for machine they aren't. The `3` is used here in the symbolic sence to represent `E`

## Most Common Application of AI

### Natural Language Processing
Humans have the most advanced method communication, known as natual language. You might have seen machines sending text and audio to the user, but actually they don't know how to process it.

NLP is the subset of AI that enables the computers to understand the meaning of what humans say. It does this by deconstructing the text gramatically, relationally and structurally. NLP leverages contexts to avoid ambiguous meaning of words.
### Speech Synthesis
The neural network model takes in several amount of audio files to get the data and then trains the generator somehow it outputs the same audio signals, thus synthesizing the voice of the human.

### Computer Vision
It's the part of AI that focuses on parts replicating the complexity of human vision system enabling machines to process and identify images and video in the same way humans do. 

It plays a vital role in facial recoganition and determination of surroundings in self-driving car

# Issues and Concerns
There are some issues and concerns related to AI that could make it some challenging.

The most prominent concern is _privacy_, that information of the customers is safe and can not be misused. 

Another is ethics and responsibility. We are using AI for limited works, but what if it is set free and it does something which is against laws or something. Who will be responsible and what ethics should it follow, because at the end its a machine, who will own his mistakes comes the difficult question.

And now comes the sci-fi part, what if we integrate it with our defence system and it turn out be human race terminator.


Isn't it seem terrifying? The AI is a tool created by a human for its help. Now it all depends on humans how it uses the tools

## AI and Ethics
Machine learning can be used for good and bad at the same time. But we can use it at the same time to counter the misuse of the technology. For example, fire is good for cooking food, but when it used to for burning people house, it's a misuse of the technology and tool. At same time, the people could use it to chase the people who are setting up the houses on fire, for good. 

The AI we are using and will be using for many decades is narrow/weak AI, it works on singularity. Which means an AI trained for playing chess won't stand up against humans. And even general AI will not, unless it's being asked to or trained to. 

AI is like a small kid, who learn from elders. Here elders are we humans, the most evolved and advanced that AI. Like other kids, at some time AI will start learning from humans. So, ethics is not a technological problem, it's more likely to be a human problem. It's something that all of us need to care about.

Let's take another example. <br>
Suppose there is some situation, a car could save either the person sitting in or the pedestrian. The company who has trained the car is accountable to all the decisions made by it.

## AI and Bias
AI is completely driven on data. The more data is, the higher will be its accuracy. Suppose the AI has been trained to detect the criminal, it will search the pictures and also the facial expression and other features that could relate to a criminal's face. Or recoganizing people based on demographic regions. 

Thus this bias in the model should be adjusted so as to make a unbiased and general model rather than strict model. However, machine learning is itself inherintly biased, which means, it works off of the fundamental assumption of bias. It bases certain input data to map them with other output data points.

As we humans, consiously prevent the biased output. Somehow subconsiously we end up outputing a biased data. When taken into large quantity of data, it can affect the model. Thus changing the dataset or limit of dataset couldn't be very helpful here.

AI is extremely powerful and can be used in ways that negatively affect society. To develop and use AI systems responsibly, AI developers must consider the ethical issues inherent in AI. They must have a realistic view of their systems and their capabilities, and be aware of the different forms of bias potentially present in their systems. With this awareness, developers can avoid unintentionally creating AI systems that have a negative rather than positive impact. Even outside the realm of science

## Making AI Trustworthy
There are 4 main aspects for the developers to make the AI system more trustworthy
1. **Transparency** &rarr; People should be aware when they are interacting with an AI system and understand 
2. **Accountability** &rarr; Developers should create AI systems with algorithmic accountability, so that any unexpected results can be traced and undone if required. 
3. **Privacy** &rarr; Personal information should always be protected and not misused by the developer.
4. **Lack of Bias** &rarr; Developers should use representative data to avoid bias, and should also audit the model regularily to adjust bias

## Jobs and AI
Like every new technology that affected some industry and jobs, AI could also automate or even replace with some manly jobs like  ice pickers who were trying to acquire large blocks of ice cubes to bring it to towns and homes so they could refrigerate their food. Those jobs were replaced by the refrigerator.

It's both a responsibility of corporations to be able to accommodate the changes in the industry and it's also the workers who have to be cognizant that some jobs may be prone to be automated and to never stop learning.

## Employment and AI
As you know, with bad good always comes, AI is most effective for job which is more repetative or rules based. For example, bank teller, call centers, salesman, drivers and etc. 

People may be fired from the job, but they can help in development of AI, like the technical people could help in designing the architecture or writing the code, and the other non-technical in perspective of AI can develop the training data. For example, making images for the car so that it can learn from it and can perform accordingly in production, etc. 

**Some points related to this**
+ AI is augmenting jobs rather than eliminating them
+ bringing new jobs to less developed countries
+ bringing investors and small business together regardless of locationwe

# The Future of AI

All the industries are being swamped by the tsunami of unstructured data, that can be multimedia, images, user details, sales etc. It is really necessary to under the data that is becoming critical. One of the most valuable applications of cognitive computing is in the health domain. 

The medical literature increases by about 700  thousand articles every year and there's already millions of journal articles out there. Now imagine a doctor and team of doctor going though it every day. This is nothing but the time and effort wasting. What if a machine could do this for us and doctor just have to input some symptoms/data to get the desired results out of that. AI in the modern world is not to try and develop a system that completely, autonomously handles every aspect of a problem but have a collaboration between machines doing what they do best and people doing what they do best. This collaborative approach could solve any upcoming problems.

## Advice for Career in AI
The field of AI is continuously growing and we have not yet discovered enough.  It's important to keep an open mind and not get too attached to any particular technology, any particular technique. 

You no longer require to pursue higher studies in mathematics or know some obscure programming language. You just have to know how to use the software API's and understand the problem. However understanding the mathematics and science behind them is equally important so that your fundamentals are clear.
