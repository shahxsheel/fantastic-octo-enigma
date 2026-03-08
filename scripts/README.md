# Scripts

## `jwt-generator.js`

### **LICENSE:** Restricted, (c) 2025 Aaron Ma, for the Satyrn internal project.

- Generates the JWT token for Sign in with Apple

```js
const privateKey = fs.readFileSync(""); // <-- fill in private key path (key.p8)
const teamId = ""; // <-- fill in Xcode team ID
const serviceId = ""; // <-- fill in service ID (com.example.app)
const keyId = ""; // <-- fill in key ID (check Certificates, Identifiers & Profiles > Keys)
```
