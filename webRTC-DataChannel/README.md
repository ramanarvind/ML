# Real time communication with WebRTC

## Introduction
The base for this PoC is the "Real time communication with WebRTC" tutorial here [https://codelabs.developers.google.com/codelabs/webrtc-web/#0]. Specifically, this PoC modifies ch.9 of the PoC where the intention is to capture an image frame from an online video and sent it over a P2P data-channel. Instead of modifying an image, in this PoC I attempted to sent text data from one browser to the other. 

## How to run the demo
Before you can run the demo, you will need to install the Javascript dependencies. Simply run the following command from your working directory:
`npm install`
Once installed, start the Node.js server by calling the following command from your work directory:
`node index.js`
The app will create a random room ID and add that ID to the URL. Open the URL from the address bar in a new browser tab or window. The demo has been tested to work on Chrome, Edge and Firefox. It is possible for the two peers to be on two completely different browsers.

## To Do
- [] Host the Node.js server on a publically accessible server and test