var express = require('express');
var router = express.Router();

const controller = require('../controller/composeController')

router.get('/', async (req, res) => {
    res.render("composer");
});

module.exports = router;