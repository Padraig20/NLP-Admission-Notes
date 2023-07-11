import {Component, ElementRef, Renderer2} from '@angular/core';
import {ToastrService} from 'ngx-toastr';
import {Router} from '@angular/router';
import {AnalyzerService} from '../../services/analyzer.service';
import {DomSanitizer} from '@angular/platform-browser';

@Component({
  selector: 'app-register-patient',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  analyzing = false;
  analyzed = false;
  text = '';
  analyzedText = '';
  constructor(
    private notification: ToastrService,
    private router: Router,
    private analyzerService: AnalyzerService,
    private renderer: Renderer2, private elementRef: ElementRef,private sanitizer: DomSanitizer
  ) {
  }

  public buttonstyleAdmission(): string {
    if (this.text === '') {
      return 'bg-gray-400';
    } else {
      return 'transition ease-in-out delay-100 duration-300 bg-blue-500 '
        + 'hover:-translate-y-0 hover:scale-110 hover:bg-blue-400 hover:cursor-pointer';
    }
  }

  sanitizeHTML(htmlString: string): string {
    const nonSpanTagsRegex = /<(?!\/?span\b)[^>]+>/gi;
    return htmlString.replace(nonSpanTagsRegex, '');
  }

  analyze() {
    this.analyzing = true;
    this.analyzerService.analyzeNote(this.text).subscribe({
      next: data => {
        this.analyzedText = '';
        console.log(data);
        const tokens = data.tokens[0];
        const entities = data.entities[0];
        let tmp = '';
        for(let i = 0; i < tokens.length; i++) {
          let token = tokens[i];
          let entity = entities[i];
          console.log(token)
          console.log(entity)
          if (entity === 'O') {
            this.analyzedText = this.analyzedText + (token + ' ');
          } else if (entity === 'B-DIAGNOSIS') {
            if ((i+1) < tokens.length && entities[i+1] === 'I-DIAGNOSIS') {
              tmp = token;
            } else {
              this.analyzedText = this.analyzedText + '<span class="fw-bold bg-green-300 text-green-800 rounded px-1 py-1 m-1">' + (token) + '<span class="fw-light ml-2 text-green-950">DIAGNOSIS</span></span> ';
              tmp = '';
            }
          } else if (entity === 'I-DIAGNOSIS') {
            tmp = tmp + (' ' + token);
            if ((i+1) < tokens.length && entities[i+1] !== 'I-DIAGNOSIS' || i === tokens.length) {
              this.analyzedText = this.analyzedText + '<span class="fw-bold bg-green-300 text-green-800 rounded px-1 py-1 m-1">' + (tmp) + '<span class="fw-light ml-2 text-green-950">DIAGNOSIS</span></span> ';
              tmp = '';
            }
          } else if (entity === 'B-AGE') {
            if ((i+1) < tokens.length && entities[i+1] === 'I-AGE') {
              tmp = token;
            } else {
              this.analyzedText = this.analyzedText + '<span class="fw-bold bg-blue-400 text-blue-900 rounded px-1 py-1 m-1">' + (token) + '<span class="fw-light ml-2 text-blue-9502">AGE</span></span> ';
              tmp = '';
            }
          } else if (entity === 'I-AGE') {
            tmp = tmp + (' ' + token);
            if ((i+1) < tokens.length && entities[i+1] !== 'I-AGE' || i === tokens.length) {
              this.analyzedText = this.analyzedText + '<span class="fw-bold bg-blue-400 text-blue-900 rounded px-1 py-1 m-1">' + (tmp) + '<span class="fw-light ml-2 text-blue-950">AGE</span></span> ';
              tmp = '';
            }
          } else if (entity === 'B-GENDER') {
            if ((i+1) < tokens.length && entities[i+1] === 'I-GENDER') {
              tmp = token;
            } else {
              this.analyzedText = this.analyzedText + '<span class="fw-bold bg-yellow-300 text-yellow-900 rounded px-1 py-1 m-1">' + (token) + '<span class="fw-light ml-2 text-yellow-950">GENDER</span></span> ';
              tmp = '';
            }
          } else if (entity === 'I-GENDER') {
            tmp = tmp + (' ' + token);
            if ((i+1) < tokens.length && entities[i+1] !== 'I-GENDER' || i === tokens.length) {
              this.analyzedText = this.analyzedText + '<span class="fw-bold bg-yellow-300 text-yellow-900 rounded px-1 py-1 m-1">' + (tmp) + '<span class="fw-light ml-2 text-yellow-950">GENDER</span></span> ';
              tmp = '';
            }
          } else if (entity === 'B-NEGATIVE') {
            if ((i+1) < tokens.length && entities[i+1] === 'I-NEGATIVE') {
              tmp = token;
            } else {
              this.analyzedText = this.analyzedText + '<span class="fw-bold bg-red-300 text-red-900 rounded px-1 py-1 m-1">' + (token) + '<span class="fw-light ml-2 text-red-950">NEGATIVE</span></span> ';
              tmp = '';
            }
          } else if (entity === 'I-NEGATIVE') {
            tmp = tmp + (' ' + token);
            if ((i+1) < tokens.length && entities[i+1] !== 'I-NEGATIVE' || i === tokens.length) {
              this.analyzedText = this.analyzedText + '<span class="fw-bold bg-red-300 text-red-900 rounded px-1 py-1 m-1">' + (tmp) + '<span class="fw-light ml-2 text-red-950">NEGATIVE</span></span> ';
              tmp = '';
            }
          }
        }
        this.notification.info('Successfully analyzed admission note!');
        console.log(this.analyzedText)
        this.analyzed = true;
        this.analyzing = false;
      },
      error: error => {
        console.log('Error analyzing note: ' + error);
        this.notification.error(error.error.message, 'Error analyzing note');
      }
    });
  }
}
