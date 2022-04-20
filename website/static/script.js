function hideFunction() {
  var x = document.getElementById("pollPartTwo");
  x.style.display = "none";
  sessionStorage.setItem("questions", x.style.display);
  var x = document.getElementById("list-example");
  x.style.display = "none";
  sessionStorage.setItem("bar", x.style.display);
  var x = document.getElementById("form-submit-info");
  x.style.display = "block";
  sessionStorage.setItem("button", x.style.display);
}

function showFunction() {
  var x = document.getElementById("pollPartTwo");
  x.style.display = "block";
  sessionStorage.setItem("questions", x.style.display);
  var x = document.getElementById("list-example");
  x.style.display = "block";
  sessionStorage.setItem("bar", x.style.display);
  var x = document.getElementById("form-submit-info");
  x.style.display = "none";
  sessionStorage.setItem("button", x.style.display);
}

function checkShowHide() {
  var x = document.getElementById("pollPartTwo");
  x.style.display = sessionStorage.getItem("questions");
  var x = document.getElementById("list-example");
  x.style.display = sessionStorage.getItem("bar");
  var x = document.getElementById("form-submit-info");
  x.style.display = sessionStorage.getItem("button");
}
