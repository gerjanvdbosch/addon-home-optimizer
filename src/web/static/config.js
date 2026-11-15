// The config form, built from the pydantic schema of domain.config.Config, so
// it follows the model without a field listed twice. Every editor is
// {node, get, compound}: get(path) reads the value (undefined when left
// empty, so the model's default applies) and tags its inputs with their path,
// which is where the server's validation errors are shown.

const schema = JSON.parse(document.getElementById("schema").textContent);
const initial = JSON.parse(document.getElementById("value").textContent);

const VARIANTS = {
    SensorReference: "Sensor",
    ComfortTargets: "Per mode",
    Tariff: "Weekdays / weekend",
};

function h(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
        if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
        else if (value != null && value !== false) node.setAttribute(key, value === true ? "" : value);
    }
    node.append(...children.filter((child) => child != null));
    return node;
}

function resolve(s) {
    if (!s.$ref) return s;
    const name = s.$ref.split("/").pop();
    return { ...schema.$defs[name], ...s, $ref: undefined, name };
}

function label(key) {
    const text = String(key).replace(/_/g, " ").replace(/\bdhw\b/i, "DHW").replace(/\bm2\b/, "m²");
    return text.charAt(0).toUpperCase() + text.slice(1);
}

function tag(node, path) {
    node.dataset.path = JSON.stringify(path);
}

function button(text, cls, onclick) {
    return h("button", { type: "button", class: cls, onclick }, text);
}

function editor(s, value) {
    s = resolve(s);
    if (s.anyOf) return union(s, value);
    if (s.name === "SensorReference") return sensor(value);
    if (s.type === "object") return object(s, value);
    if (s.type === "array" && s.prefixItems) return tuple(s, value);
    if (s.type === "array" && resolve(s.items).prefixItems) return schedule(resolve(s.items), value);
    if (s.type === "array") return list(s, value);
    return scalar(s, value);
}

function scalar(s, value) {
    value ??= s.default;
    if (s.type === "boolean") {
        const input = h("input", { type: "checkbox", checked: !!value });
        return { node: input, get: (path) => (tag(input, path), input.checked) };
    }
    const number = s.type === "number" || s.type === "integer";
    const time = s.format === "time";
    const input = h("input", {
        type: time ? "time" : number ? "number" : "text",
        step: s.type === "integer" ? 1 : number ? "any" : null,
        min: s.minimum,
        max: s.maximum,
        placeholder: s.title,
        value: time && value ? value.slice(0, 5) : value,
    });
    return {
        node: input,
        input,
        get: (path) => {
            tag(input, path);
            const text = input.value.trim();
            if (text === "") return undefined;
            return number ? Number(text) : text;
        },
    };
}

function sensor(value) {
    const entity = h("input", { type: "text", class: "entity", placeholder: "entity_id", value: value?.entity_id });
    const attribute = h("input", { type: "text", placeholder: "attribute", value: value?.attribute });
    return {
        node: h("div", { class: "inline" }, entity, attribute),
        get: (path) => {
            // A missing sensor is reported at its own path.
            tag(entity, path);
            tag(attribute, [...path, "attribute"]);
            // No entity: the sensor is not configured.
            if (!entity.value.trim()) return undefined;
            return { entity_id: entity.value.trim(), attribute: attribute.value.trim() || null };
        },
    };
}

function tuple(s, value) {
    const parts = s.prefixItems.map((item, i) => scalar(item, value?.[i]));
    return {
        node: h("div", { class: "inline" }, ...parts.map((part) => part.node)),
        get: (path) => {
            const values = parts.map((part, i) => part.get([...path, i]));
            return values.every((v) => v === undefined) ? undefined : values;
        },
    };
}

function field(key, e) {
    // A choice lays out its own label.
    if (e.label) return e.label(label(key)), e.node;
    if (!e.compound) return h("div", { class: "row" }, h("span", {}, label(key)), e.node);
    return h("div", { class: "group" }, h("div", { class: "heading" }, label(key)), e.node);
}

// A top-level panel, as a Home Assistant card.
function panel(key, body) {
    return h("section", { class: "card" }, h("h2", {}, label(key)), body);
}

function object(s, value, root = false) {
    value ??= {};
    const editors = Object.entries(s.properties).map(([key, p]) => [key, editor(p, value[key])]);
    const node = h("div", { class: "fields" });
    const general = h("div", { class: "fields" });
    // Groups stand flat, under a heading, so a field after one would read as
    // its own: an object's fields come first, its groups after them.
    const groups = [];
    for (const [key, e] of editors) {
        const p = resolve(s.properties[key]);
        // The top level's loose fields go together in one panel.
        const loose = p.type !== "object" || p.name === "SensorReference";
        if (!root && e.compound) groups.push(field(key, e));
        else if (!root) node.append(field(key, e));
        else if (loose) general.append(field(key, e));
        else node.append(panel(key, e.node));
    }
    node.append(...groups);
    if (root && general.children.length) node.prepend(panel("general", general));
    return {
        node: root ? node.childNodes : node,
        compound: true,
        get: (path) => {
            tag(node, path);
            const out = {};
            for (const [key, e] of editors) {
                const v = e.get([...path, key]);
                if (v !== undefined) out[key] = v;
            }
            return out;
        },
    };
}

function list(s, value) {
    const items = [];
    const body = h("div");
    const add = (v) => {
        const e = editor(s.items, v);
        const item = { e };
        item.node = h("div", { class: "item" }, e.node, button("✕", "remove", () => {
            item.node.remove();
            items.splice(items.indexOf(item), 1);
        }));
        items.push(item);
        body.append(item.node);
    };
    (value ?? []).forEach(add);
    return {
        node: h("div", { class: "fields" }, body, button("+ Add", "add", () => add())),
        // A list of single fields stands in the fields' column, as one field.
        compound: editor(s.items).compound,
        get: (path) => {
            const out = [];
            for (const item of items) {
                const v = item.e.get([...path, out.length]);
                if (v !== undefined) out.push(v);
            }
            return out;
        },
    };
}

function nextHour(clock) {
    const [hour, minute] = clock.split(":");
    return `${String((Number(hour) + 1) % 24).padStart(2, "0")}:${minute}`;
}

// A daily schedule: rows of (from, values...), kept in time order.
function schedule(item, value) {
    const columns = item.prefixItems;
    const rows = [];
    const body = h("tbody");
    const add = (v) => {
        const cells = columns.map((c, i) => scalar({ ...c, title: undefined }, v?.[i]));
        const row = { cells };
        row.node = h("tr", {}, ...cells.map((c) => h("td", {}, c.node)), h("td", {}, button("✕", "remove", () => {
            row.node.remove();
            rows.splice(rows.indexOf(row), 1);
        })));
        rows.push(row);
        body.append(row.node);
    };
    (value ?? []).forEach(add);
    // A new row continues from the last: an hour later, the same values.
    const next = () => {
        const last = rows.at(-1)?.cells.map((c) => c.input.value);
        if (last?.[0]) last[0] = nextHour(last[0]);
        add(last?.map((v) => (v === "" ? undefined : v)));
    };
    const head = h("tr", {}, ...columns.map((c) => h("th", {}, c.title ?? (c.format === "time" ? "From" : "Value"))), h("th"));
    const node = h("div", { class: "schedule fields" }, h("table", {}, h("thead", {}, head), body), button("+ Add", "add", next));
    return {
        node,
        get: (path) => {
            rows.sort((a, b) => a.cells[0].input.value.localeCompare(b.cells[0].input.value));
            rows.forEach((row) => body.append(row.node));
            tag(node, path);
            // No rows: none set (a mode without one is not planned).
            if (!rows.length) return undefined;
            return rows.map((row, i) => row.cells.map((c, j) => c.get([...path, i, j]) ?? null));
        },
    };
}

function variant(s) {
    if (s.name) return VARIANTS[s.name] ?? label(s.name);
    if (s.type === "array") return "Schedule";
    if (s.type === "number" || s.type === "integer") return "Fixed";
    return label(s.type);
}

function fits(s, value) {
    if (value == null) return s.type === "null" ? 1 : 0;
    if (Array.isArray(value)) return s.type === "array" ? 1 : 0;
    if (typeof value === "number") return s.type === "number" || s.type === "integer" ? 1 : 0;
    if (typeof value !== "object") return s.type === typeof value ? 1 : 0;
    if (s.type !== "object" || !(s.required ?? []).every((key) => key in value)) return 0;
    return 1 + Object.keys(value).filter((key) => key in s.properties).length;
}

function union(s, value) {
    value ??= s.default;
    const options = s.anyOf.map(resolve);
    const nullable = options.some((o) => o.type === "null");
    const choices = options.filter((o) => o.type !== "null");

    if (choices.length === 1) {
        const only = choices[0];
        // An optional group: switched on or off as a whole.
        if (only.type === "object" && only.name !== "SensorReference") {
            const box = h("input", { type: "checkbox", checked: value != null });
            const e = editor(only, value);
            const body = h("div", { hidden: value == null }, e.node);
            box.addEventListener("change", () => (body.hidden = !box.checked));
            return {
                node: h("div", { class: "fields" }, h("div", { class: "row" }, h("span", {}, "Enabled"), box), body),
                compound: true,
                get: (path) => (box.checked ? e.get(path) : null),
            };
        }
        // Anything else optional is unset when left empty.
        const e = editor(only, value);
        return { ...e, get: (path) => e.get(path) ?? (nullable ? null : undefined) };
    }

    const scores = choices.map((c) => fits(c, value));
    let chosen = value == null && nullable ? -1 : scores.indexOf(Math.max(...scores));
    const editors = new Map();
    const body = h("div");
    const select = h("select", {}, ...(nullable ? [h("option", { value: -1 }, "None")] : []),
        ...choices.map((c, i) => h("option", { value: i, selected: i === chosen }, variant(c))));
    const show = () => {
        // Each variant keeps its own edits while another is shown.
        if (chosen >= 0 && !editors.has(chosen)) {
            editors.set(chosen, editor(choices[chosen], fits(choices[chosen], value) ? value : undefined));
        }
        const e = editors.get(chosen);
        body.replaceChildren(...(e ? [e.node] : []));
        // A field or a schedule stands in a row, under the choice; a
        // variant of groups below the choice's heading.
        node.classList.toggle("side", !e || !e.compound);
    };
    const name = h("span", { class: "label" });
    const node = h("div", { class: "choice" }, h("div", { class: "pick" }, name, h("div", { class: "row" }, h("span", {}, "Type"), select)), body);
    select.addEventListener("change", () => ((chosen = Number(select.value)), show()));
    show();
    return {
        node,
        compound: true,
        label: (text) => (name.textContent = text),
        get: (path) => {
            tag(select, path);
            return chosen < 0 ? null : editors.get(chosen).get(path);
        },
    };
}

// Validation errors carry the field's path, with the tried union variant's
// name between its parts: an input matches when its path runs through them.
function matches(path, loc) {
    let i = 0;
    for (const part of loc) if (i < path.length && String(path[i]) === String(part)) i++;
    return i === path.length;
}

const within = (inner, outer) => outer.every((part, i) => String(inner[i]) === String(part));

function showErrors(errors) {
    const tagged = [...form.querySelectorAll("[data-path]")].map((node) => [node, JSON.parse(node.dataset.path)]);
    const found = errors.map((error) => {
        const loc = error.loc.slice(1); // past "body"
        const best = tagged.filter(([, path]) => matches(path, loc)).sort((a, b) => b[1].length - a[1].length)[0];
        return { error, loc, node: best?.[0] ?? form, path: best?.[1] ?? [] };
    });
    // Deepest first: a union reports every variant it tried, and only the
    // one with an error inside the field shown says what is wrong.
    found.sort((a, b) => b.path.length - a.path.length);
    const shown = [];
    for (const { error, loc, node, path } of found) {
        if (path.length && shown.some((other) => within(other, path))) continue;
        shown.push(path);
        node.classList.add("invalid");
        const anchor = node.closest(".row, .schedule") ?? node;
        anchor.insertAdjacentElement("afterend", h("div", { class: "error" }, `${loc.join(" › ")}: ${error.msg}`));
    }
    form.querySelector(".error")?.scrollIntoView({ block: "center" });
    return shown.length;
}

function setStatus(text, failed = false) {
    statusText.textContent = text;
    statusText.classList.toggle("failed", failed);
}

async function save(event) {
    event.preventDefault();
    form.querySelectorAll(".error").forEach((node) => node.remove());
    form.querySelectorAll(".invalid").forEach((node) => node.classList.remove("invalid"));
    setStatus("Saving…");

    const response = await fetch("api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(root.get([])),
    });
    if (response.status === 422) {
        const count = showErrors((await response.json()).detail);
        return setStatus(`${count} field${count === 1 ? "" : "s"} to fix`, true);
    }
    if (!response.ok) return setStatus(`Save failed (${response.status})`, true);

    // The worker writes the file in its turn, after any job it is running.
    const { job_id } = await response.json();
    while ((await (await fetch("api/status")).json()).some((job) => job.id === job_id)) {
        setStatus("Waiting for the worker…");
        await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    setStatus("Saved");
}

const form = document.getElementById("config");
const statusText = document.getElementById("status");
const root = object(schema, initial, true);
form.append(...root.node);
form.addEventListener("submit", save);
