### IRReC search engine frontend

This readme briefly describes how to run the frontend tool locally. This User Interface (UI) was prepared by Lee Maguire for BIMacademy.

**Figure:** Example screenshot of the UI:
![alt text](https://github.com/rubenkruiper/irrec/blob/main/demonstrator.jpeg?raw=true)

## Install

1. You will require node.js to be installed to the system to continue https://nodejs.org
2. In a terminal or powershell run the command `npm i`
3. When done run `npm run dev` to run a local instance on your machine.

Running a local instance is enough to demo the system. W.r.t. configuration, mostly standard [Vite](https://vitejs.dev/) ports are used. When setting the port to which the UI listens, modify `**/UserInterface/src/store/SearchStore.ts`. I've been running the following commands:

* `ssh -L 8000:localhost:8503 \[user\]@\[external.IP\]`
Inside UI folder (currently set to listen to port 8000, and host at 8080):
* `npm run dev` 
Then in a browser go to http://localhost:8080/ to open the app.

<!-- ## Build 

1. Complete steps 1 - 3 above first
2. Run the command `npm run build`
3. The compiled frontend will appear in a `dist` folder in the current folder

 -->
