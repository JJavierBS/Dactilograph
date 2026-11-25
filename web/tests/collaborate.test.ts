import { describe, expect, it } from "vitest";
import { COLLABORATION_EMAIL, collaborationMailto } from "../src/collaborate";

function parse(mailto: string): { address: string; subject: string; body: string } {
  const [address, query] = mailto.replace("mailto:", "").split("?");
  const params = new URLSearchParams(query);
  return { address: address!, subject: params.get("subject")!, body: params.get("body")! };
}

describe("collaborationMailto", () => {
  it("fills the template with the form fields", () => {
    const mail = parse(
      collaborationMailto({
        name: " Ana ",
        profile: "Intérprete de LSE",
        letters: "Ñ, RR",
        message: "La Ñ se hace con otro movimiento & más.",
      }),
    );
    expect(mail.address).toBe(COLLABORATION_EMAIL);
    expect(mail.subject).toBe("Colaborar con Dactilograph: Ana");
    expect(mail.body).toContain("Nombre: Ana\r\n");
    expect(mail.body).toContain("Mi relación con la LSE: Intérprete de LSE");
    expect(mail.body).toContain("se pueden mejorar: Ñ, RR");
    expect(mail.body).toContain("otro movimiento & más.");
  });

  it("uses dashes and a generic subject when fields are empty", () => {
    const mail = parse(collaborationMailto({ name: "", profile: "", letters: " ", message: "" }));
    expect(mail.subject).toBe("Quiero colaborar con Dactilograph");
    expect(mail.body).toContain("Nombre: -");
    expect(mail.body).toContain("Comentarios:\r\n-");
  });
});
