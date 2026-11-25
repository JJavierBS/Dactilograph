export const COLLABORATION_EMAIL = "josejavierbravosarabia@gmail.com";
export const COLLABORATE_ACTION = "colabora";
export const COLLABORATION_SECTION_ID = "colabora";

export interface CollaborationRequest {
  readonly name: string;
  readonly profile: string;
  readonly letters: string;
  readonly message: string;
}

function orDash(value: string): string {
  return value.trim() === "" ? "-" : value.trim();
}

export function collaborationMailto(request: CollaborationRequest): string {
  const name = request.name.trim();
  const subject =
    name === "" ? "Quiero colaborar con Dactilograph" : `Colaborar con Dactilograph: ${name}`;
  const body = [
    "Hola:",
    "",
    "Me gustaría ayudar a mejorar los gestos de Dactilograph.",
    "",
    `Nombre: ${orDash(request.name)}`,
    `Mi relación con la LSE: ${orDash(request.profile)}`,
    `Letras que creo que están mal o se pueden mejorar: ${orDash(request.letters)}`,
    "",
    "Comentarios:",
    orDash(request.message),
    "",
    "Puedo grabar mis propios ejemplos con la aplicación de escritorio para mejorar los datos.",
  ].join("\r\n");
  return `mailto:${COLLABORATION_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
}

function field(data: FormData, name: string): string {
  const value = data.get(name);
  return typeof value === "string" ? value : "";
}

export function bindCollaborationForm(form: HTMLFormElement): void {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const data = new FormData(form);
    window.location.href = collaborationMailto({
      name: field(data, "nombre"),
      profile: field(data, "perfil"),
      letters: field(data, "letras"),
      message: field(data, "mensaje"),
    });
  });
}

export function scrollToCollaboration(): void {
  document
    .getElementById(COLLABORATION_SECTION_ID)
    ?.scrollIntoView({ behavior: "smooth", block: "start" });
}
