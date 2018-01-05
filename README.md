...In Progress...
<br>
<h3>Model Deployment Strategies</h3>
<br>As Data Scientists, we spend so much time exploring the data, working on transformations, enriching our core data with external sources, all the work in between to prepare our model pipeline, and then eventually training a machine learning model (whether a regression model, classifier, clustering algorithm, etc.)
<br>
<br>At this point, you have a powerful and extremely valuable model...but it's worthless if it is not deployed in production within your organization. 
<br>
<br>Just as important, you need deploy your entire model pipeline into production. This takes into consideration all of the data tranformations, enrichments, and predictive models required to take your data from raw to enriched/scored data within you database.
<br>This repo contains a few of my ideas, strategies, and code for deploying your models in production and at scale.
<br>
<br><h3>So where can machine learning models be deployed?</h3>
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy as a Batch Process
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy as a Web Service (REST API)
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy within Web App
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy online as part of a real-time data stream
<br>&nbsp;&nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;Deploy/Embed within Devices
<br>
<br><h3>Based on these deployments, here are high-level flows for several deployment designs (including open source tech being used): </h3>
NOTE: A deeper-dive design ref architecture is in progress for each
<br>
<br><b>ML Deployment Design - Batch</b>
<br><img src="images/ml_deployment_batch.png" class="inline"/>
<br>
<br><b>ML Deployment Design - Batch</b>
<br><img src="images/ml_deployment_batch_dsx_hdp_livy.png" class="inline"/>
<br>
<br><b>ML Deployment Design - Batch</b>
<br><img src="images/ml_deployment_batch_dsx_hdp.png" class="inline"/>
<br>
<br><b>ML Deployment Design - API Endpoint</b>
<br><img src="images/ml_deployment_dsx_api.png" class="inline"/>
<br>
<br><b>ML Deployment Design - Real-time, Streaming Analytics</b>
<br><img src="images/ml_deployment_streaming_sam.png" class="inline"/>
<br>
<br><b>References:</b>
<br>
