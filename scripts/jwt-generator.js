/// (c) 2025 Aaron Ma, for the Satyrn internal project.
/// Restricted license.

const jwt = require("jsonwebtoken");
const fs = require("fs");

const privateKey = fs.readFileSync("key.p8"); // <-- fill in private key path (key.p8)
const teamId = ""; // <-- fill in Xcode team ID
const serviceId = ""; // <-- fill in service ID (com.example.app)
const keyId = ""; // <-- fill in key ID (check Certificates, Identifiers & Profiles > Keys)

const claims = {
  iss: teamId,
  iat: Math.floor(Date.now() / 1000),
  exp: Math.floor(Date.now() / 1000) + 86400 * 180, // 6 months
  aud: "https://appleid.apple.com",
  sub: serviceId,
};

const token = jwt.sign(claims, privateKey, {
  algorithm: "ES256",
  keyid: keyId,
});

console.log(token);
