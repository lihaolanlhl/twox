import { Component, OnInit } from '@angular/core';
import { Twox } from './twox';
import {HttpClient} from '@angular/common/http';
import {RequestOptions} from '@angular/http';
import { FlexLayoutModule } from '@angular/flex-layout';

@Component({
  selector: 'app-twox',
  templateUrl: './twox.component.html',
  styleUrls: ['./twox.component.css']
})
export class TwoxComponent implements OnInit {
  title = 'Twox';
  twox = new Twox('', this.http);
  constructor(private http: HttpClient) { }
  // constructor(private http: HttpClient) {}
  ngOnInit() {
  }
}
