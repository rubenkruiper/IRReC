const express = require("express");
const app = express();

app.use(express.static(__dirname + "/dist"));

// app.get("/", (req, res) => {
//   res.render("dist/index");
// });

app.listen(8000);
