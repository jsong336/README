# ForMe
What's best financial advice FOR ME?

## Objective
People often ask, what's the best financial option FOR ME?
Our end product goal is to be TD based customer's go-to mobile application to receive financial advice based on their spending habits

Streamline the process of informing the customers of better financial choices by using select machine learning algorithms to analyze the TD's big data on customers' data and their transaction histories.

## Feature Provided
<ul>
   <li>Credit Card Type Recommendation</li>
   <li>Account Types Recommendation</li>
   <li>Closest Branch Location</li>
   <li>Information Regarding Each (Account/ Credit) types</li>
</ul>

## Process
## Parameters
<li>age</li>
<li>income</li>
<li>food and dining</li>
<li>shopping</li>
<li>home</li>
<li>entertainment</li>
<li>fees and charges</li>
<li>food and dining count</li>
<li>shopping count</li>
<li>home count</li>
<li>entertainment count</li>
<li>fees and charges count</li>

## K-MEANS Algorithms & silhouette

Silhouette scores around 0.55 – 0.57 when K = 5 ~ 6
Silhouette scores: how close each points are in each cluster

## Clusters output 
Total payment: total spent for each of the categories from transaction
Frequency: how often made the transaction
Increasing frequency and total payment of food and dining as income increase
Green, Yellow, Grey: Higher income compared to blue or red


### Analysis
Blue & Red: makes one time pays for home 
Green & Yellow & Grey: frequent payments for home
Greys & Parts of Green: frequently shopping as well as high total payment
Blue: does not spend much of money on shopping


Yellow: frequent but small fees and charge transaction
Green &Black &Yellow: payments to entertainment
Blue & Red: does not pay for any entertainment 

## Classes to Card/Account




### Setup on AWS - C9
```
touch mongodb-org-3.6.repo
```
Open mongodb-org-3.6.repo and save below code
```
[mongodb-org-3.6]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/amazon/2013.03/mongodb-org/3.6/x86_64/
gpgcheck=1
enabled=1
gpgkey=https://www.mongodb.org/static/pgp/server-3.6.asc
```
```
sudo mv mongodb-org-3.6.repo /etc/yum.repos.d
sudo yum install -y mongodb-org
```
```
mkdir datda
echo 'mongod --dbpath=data --nojournal' > mongod
chmod a+x mongod
```
Run ``` ./mongod ```


## Understanding the Basic Structure and Flow of Program
***Flow List***
1. User logs into the app
2. Login fires a GET request to the server with customer_id (either hosted locally or online)
3. Server routes requests accordingly (server setup with (**NodeJS**) and (**ExpressJS**)
   - one goes to *TD_Davinci_API_Closest_Branch.py* where a GET request is made to **TD's Davinci API** to retrieve customer's and customer's nearest branch location in geographical coordinates
   ```python
      customer_details = requests.get(
         'https://api.td-davinci.com/api/customers/'+customer_id,
         headers = { 'Authorization': API_KEY }
      )
   ```
   - the other is processed within the server to query matching customer data from the database(**mongoDB**) which holds a demo data that was generated using K-mean algorithms from (**scikit-learn**)
  ![alt text](https://raw.githubusercontent.com/hPark0811/ForMe/master/server/Tools/KMEAN/graph/Food_Dining.png)
4. Retrieved data is then presented to the app's UI to display appropriate bank account/credit card information catered to each user along with a map that displays the location of a TD branch that is nearest to the user's home address (**GOOGLE MAPS API**)

