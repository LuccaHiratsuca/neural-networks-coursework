// docs/javascripts/mathjax.js
window.MathJax = {
  tex: {
    inlineMath: [["$", "$"], ["\\(", "\\)"]],
    displayMath: [["$$", "$$"], ["\\[", "\\]"]],
    processEscapes: true,
  },
  options: {
    // arithmatex marca os trechos com a classe "arithmatex"
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};
