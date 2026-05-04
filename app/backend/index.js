const express = require("express");
const mysql = require("mysql2/promise");
require("dotenv").config();

const app = express();
app.use(express.json());

const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_DATABASE,
  waitForConnections: true,
  connectionLimit: 100,
  queueLimit: 0,
});

const cache = new Map();

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  next();
});

app.get("/search", async (req, res) => {
  console.log("search req");
  const query = req.query.q;

  if (!query) {
    return res.status(400).json({ error: "Query parameter 'q' is required" });
  }

  if (cache.has(query)) {
    console.log("Using cache for query:", query);
    return res.json(cache.get(query));
  }

  try {
    const [results] = await pool.query(
      `SELECT Title, MovieId FROM movie WHERE Title LIKE ? LIMIT 10`,
      [`${query}%`]
    );
    cache.set(query, results);
    res.json(results);
  } catch (error) {
    console.error("Error executing query", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Clear cache every 8 hours
setInterval(() => {
  cache.clear();
  console.log("Cache cleared");
}, 8000 * 60 * 60);

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
