import { brand, t, translatePage } from "./i18n.js";

translatePage();
const logo = document.getElementById("brand-logo");
if (brand.logo.url) {
  const img = document.createElement("img");
  img.src = brand.logo.url;
  img.alt = "";
  logo.replaceChildren(img, " ", brand.product_name);
} else {
  logo.textContent = `${brand.logo.text} ${brand.product_name}`;
}

const form = document.getElementById("login-form");
const error = document.getElementById("login-error");

// The form also works without this script (a plain POST /login); its errors come back
// as ?error=failed|throttled.
const failure = new URLSearchParams(location.search).get("error");
if (failure) {
  error.textContent = t(failure === "throttled" ? "auth.throttled" : "auth.failed");
  error.hidden = false;
  history.replaceState(null, "", "/login");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.hidden = true;
  const button = form.querySelector("button");
  button.disabled = true;
  try {
    const response = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: form.username.value, password: form.password.value }),
    });
    if (response.ok) {
      location.replace("/");
      return;
    }
    const body = await response.json().catch(() => ({}));
    error.textContent = typeof body.detail === "string" ? body.detail : t("auth.failed");
  } catch {
    error.textContent = t("error.server");
  } finally {
    button.disabled = false;
  }
  error.hidden = false;
  form.password.select();
});
