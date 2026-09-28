// Brand and translations, delivered by the server inside index.html (#boot).

const boot = JSON.parse(document.getElementById("boot").textContent);
const LANGUAGE_KEY = "origin:language";

function storedLanguage() {
  try {
    return localStorage.getItem(LANGUAGE_KEY);
  } catch {
    return null;
  }
}

/** The viewer's choice (Settings), else the brand's / ORIGIN_LOCALE. */
export const locale = boot.messages[storedLanguage()] ? storedLanguage() : boot.locale;
export const brand = boot.brand;
export const languages = boot.languages;
export const version = boot.version;
/** Signed-in user ({username, is_admin}) with ORIGIN_AUTH=password; null otherwise. */
export const user = boot.user;
const messages = boot.messages[locale] || {};
const fallback = boot.messages.en || {};

// Names every message may use: "Mensagem para o {assistant}…".
const globals = { product: brand.product_name, assistant: brand.assistant_name };

/** The message for `key`; plural entries pick "one"/"other" from `vars.n`. */
export function t(key, vars = {}) {
  let entry = messages[key] ?? fallback[key];
  if (entry == null) return key;
  if (typeof entry === "object" && !Array.isArray(entry)) entry = vars.n === 1 ? entry.one : entry.other;
  if (Array.isArray(entry)) return entry;
  const values = { ...globals, ...vars };
  return entry.replace(/\{(\w+)\}/g, (match, name) => (name in values ? String(values[name]) : match));
}

export function setLanguage(value) {
  try {
    localStorage.setItem(LANGUAGE_KEY, value);
  } catch {}
  location.reload();
}

/** Fill `data-i18n` (text) and `data-i18n-attr="attr:key;attr:key"` in the static HTML. */
export function translatePage(root = document) {
  document.documentElement.lang = locale;
  root.querySelectorAll("[data-i18n]").forEach((node) => (node.textContent = t(node.dataset.i18n)));
  root.querySelectorAll("[data-i18n-attr]").forEach((node) => {
    for (const pair of node.dataset.i18nAttr.split(";")) {
      const [attr, key] = pair.split(":");
      node.setAttribute(attr.trim(), t(key.trim()));
    }
  });
}

export const formatNumber = (n) => n.toLocaleString(locale);
export const formatDateTime = (date) => date.toLocaleString(locale);
